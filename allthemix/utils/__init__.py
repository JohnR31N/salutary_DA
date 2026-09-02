"""Dependency-light shared utilities.

- ``parallel``            device sharding/replication primitives
- ``checkpoint``          msgpack/orbax checkpoint IO
- ``backend_environment`` fail-closed JAX/torch-XLA environment guard
- ``reproducibility``     seed plumbing

Domain-specific utilities live with their domains: dataset helpers in
``data.datasets``, data helpers in ``data.utils``, the saliency cache in
``data.saliency``, early-stop and LR schedules in ``training``.
"""
