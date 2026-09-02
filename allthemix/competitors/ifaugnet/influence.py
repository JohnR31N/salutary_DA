from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree


def classifier_logits(
    features: jnp.ndarray,
    classifier_params: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Apply the frozen final linear classifier to feature vectors."""
    logits = features @ classifier_params["kernel"]

    if "bias" in classifier_params:
        logits = logits + classifier_params["bias"]

    return logits


def classifier_loss(
    classifier_params: dict[str, jnp.ndarray],
    features: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mean cross entropy for the final classifier layer."""
    logits = classifier_logits(
        features=features,
        classifier_params=classifier_params,
    )
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits,
        labels.astype(
            jnp.int32,
        ),
    )

    return jnp.mean(
        losses,
    )


def classifier_grad(
    classifier_params: dict[str, jnp.ndarray],
    features: jnp.ndarray,
    labels: jnp.ndarray,
) -> Any:
    """Differentiate final-layer loss with respect to final-layer params."""
    return jax.grad(
        classifier_loss,
    )(
        classifier_params,
        features,
        labels,
    )


def last_layer_grad_per_example(
    features: jnp.ndarray,
    labels: jnp.ndarray,
    classifier_params: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Return one final-layer gradient tree per example."""
    logits = classifier_logits(
        features=features,
        classifier_params=classifier_params,
    )
    residual = (  # d CE / d logits = softmax(logits) - one_hot(y).
        jax.nn.softmax(
            logits,
            axis=-1,
        )
        - jax.nn.one_hot(
            labels.astype(
                jnp.int32,
            ),
            logits.shape[-1],
        )
    )
    gradients = {
        "kernel": jnp.einsum(  # grad W_i = feature_i outer residual_i.
            "bd,bc->bdc",
            features,
            residual,
        ),
    }

    if "bias" in classifier_params:
        gradients["bias"] = residual

    return gradients


def _per_example_dot(
    per_example_gradients: dict[str, jnp.ndarray],
    vector: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Dot each example gradient with a fixed parameter-space vector."""
    dots = jnp.einsum(
        "bdc,dc->b",
        per_example_gradients["kernel"],
        vector["kernel"],
    )

    if "bias" in per_example_gradients and "bias" in vector:
        dots = dots + jnp.einsum(
            "bc,c->b",
            per_example_gradients["bias"],
            vector["bias"],
        )

    return dots


def conjugate_gradient(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    right_hand_side: jnp.ndarray,
    max_iter: int = 50,
    ridge: float = 1.0e-8,
) -> jnp.ndarray:
    """Solve A x = b with a fixed-length conjugate-gradient scan."""
    solution = jnp.zeros_like(
        right_hand_side,
    )
    residual = right_hand_side - matvec(
        solution,
    )
    direction = residual
    residual_squared = jnp.vdot(
        residual,
        residual,
    )

    def body(carry, _):
        """Run one conjugate-gradient iteration."""
        solution, residual, direction, residual_squared = carry
        matrix_direction = matvec(
            direction,
        )
        alpha = residual_squared / (
            jnp.vdot(
                direction,
                matrix_direction,
            )
            + ridge
        )
        new_solution = solution + alpha * direction
        new_residual = residual - alpha * matrix_direction
        new_residual_squared = jnp.vdot(
            new_residual,
            new_residual,
        )
        beta = new_residual_squared / (
            residual_squared + ridge
        )
        new_direction = new_residual + beta * direction

        return (
            new_solution,
            new_residual,
            new_direction,
            new_residual_squared,
        ), None

    (
        solution,
        _,
        _,
        _,
    ), _ = jax.lax.scan(
        body,
        (
            solution,
            residual,
            direction,
            residual_squared,
        ),
        xs=None,
        length=max_iter,
    )

    return solution


def compute_s_test(
    classifier_params: dict[str, jnp.ndarray],
    train_features: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_features: jnp.ndarray,
    validation_labels: jnp.ndarray,
    damping: float = 1.0e-2,
    cg_iters: int = 50,
) -> dict[str, jnp.ndarray]:
    """Estimate s_test = (H_train + damping I)^-1 grad L_validation."""
    validation_gradient = classifier_grad(
        classifier_params=classifier_params,
        features=validation_features,
        labels=validation_labels,
    )
    flat_validation_gradient, unravel = ravel_pytree(
        validation_gradient,
    )

    def flat_train_gradient(params):
        """Flatten the train-loss gradient for Hessian-vector products."""
        gradient_tree = classifier_grad(
            classifier_params=params,
            features=train_features,
            labels=train_labels,
        )
        flat_gradient, _ = ravel_pytree(
            gradient_tree,
        )

        return flat_gradient

    def damped_hessian_vector_product(flat_vector):
        """Compute (H_train + damping I) times one flat vector."""
        vector_tree = unravel(
            flat_vector,
        )
        _, hessian_vector = jax.jvp(
            flat_train_gradient,
            (classifier_params,),
            (vector_tree,),
        )

        return hessian_vector + damping * flat_vector

    flat_s_test = conjugate_gradient(
        matvec=damped_hessian_vector_product,
        right_hand_side=flat_validation_gradient,
        max_iter=cg_iters,
    )

    return unravel(
        flat_s_test,
    )


def s_test_residual_norm(
    classifier_params: dict[str, jnp.ndarray],
    train_features: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_features: jnp.ndarray,
    validation_labels: jnp.ndarray,
    s_test: dict[str, jnp.ndarray],
    damping: float = 1.0e-2,
) -> jnp.ndarray:
    """Measure the relative residual of the damped inverse-Hessian solve."""
    validation_gradient = classifier_grad(
        classifier_params=classifier_params,
        features=validation_features,
        labels=validation_labels,
    )
    flat_validation_gradient, unravel = ravel_pytree(
        validation_gradient,
    )
    flat_s_test, _ = ravel_pytree(
        s_test,
    )

    def flat_train_gradient(params):
        """Flatten train gradients for the residual Hessian-vector product."""
        gradient_tree = classifier_grad(
            classifier_params=params,
            features=train_features,
            labels=train_labels,
        )
        flat_gradient, _ = ravel_pytree(
            gradient_tree,
        )

        return flat_gradient

    _, hessian_vector = jax.jvp(
        flat_train_gradient,
        (classifier_params,),
        (unravel(flat_s_test),),
    )
    residual = (  # r = (H + damping I) s_test - grad L_validation.
        hessian_vector
        + damping * flat_s_test
        - flat_validation_gradient
    )

    return jnp.linalg.norm(
        residual,
    ) / (
        jnp.linalg.norm(
            flat_validation_gradient,
        )
        + 1.0e-12
    )


def influence_up_loss(
    features: jnp.ndarray,
    labels: jnp.ndarray,
    classifier_params: dict[str, jnp.ndarray],
    s_test: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Compute per-example influence-up values on validation loss."""
    per_example_gradients = last_layer_grad_per_example(
        features=features,
        labels=labels,
        classifier_params=classifier_params,
    )

    return -_per_example_dot(  # I_up(z) = -grad L(z)^T s_test.
        per_example_gradients=per_example_gradients,
        vector=s_test,
    )
