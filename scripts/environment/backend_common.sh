#!/usr/bin/env bash

# Shared helpers for the isolated JAX and PyTorch/XLA environments.
# This file is sourced by the public setup, switch, and check entry points.

allthemix_environment_repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/../.." && pwd
}

allthemix_state_root() {
    if [[ -n "${ALLTHEMIX_STATE_ROOT:-}" ]]; then
        printf '%s\n' "${ALLTHEMIX_STATE_ROOT}"
    elif [[ -d /mnt/disks/allthemix ]] && \
        { ! command -v mountpoint >/dev/null 2>&1 || \
          mountpoint -q /mnt/disks/allthemix; }; then
        printf '%s\n' /mnt/disks/allthemix
    else
        printf '%s\n' "${HOME}/.local/share/allthemix"
    fi
}

allthemix_backend_env() {
    local backend="$1"
    local state_root
    state_root="$(allthemix_state_root)"

    case "${backend}" in
        jax)
            printf '%s\n' \
                "${ALLTHEMIX_JAX_ENV:-${state_root}/venvs/allthemix-jax}"
            ;;
        xla)
            printf '%s\n' \
                "${ALLTHEMIX_XLA_ENV:-${state_root}/venvs/allthemix-xla}"
            ;;
        *)
            echo "Unsupported backend: ${backend}" >&2
            return 2
            ;;
    esac
}

allthemix_backend_python() {
    local backend="$1"
    local variable
    case "${backend}" in
        jax) variable=ALLTHEMIX_JAX_PYTHON ;;
        xla) variable=ALLTHEMIX_XLA_PYTHON ;;
        *)
            echo "Unsupported backend: ${backend}" >&2
            return 2
            ;;
    esac

    if [[ -n "${!variable:-}" ]]; then
        printf '%s\n' "${!variable}"
    else
        printf '%s/bin/python\n' "$(allthemix_backend_env "${backend}")"
    fi
}

allthemix_configure_storage() {
    local state_root
    state_root="$(allthemix_state_root)"

    export ALLTHEMIX_STATE_ROOT="${state_root}"
    export ALLTHEMIX_JAX_ENV="${ALLTHEMIX_JAX_ENV:-${state_root}/venvs/allthemix-jax}"
    export ALLTHEMIX_XLA_ENV="${ALLTHEMIX_XLA_ENV:-${state_root}/venvs/allthemix-xla}"
    export ALLTHEMIX_JAX_PYTHON="${ALLTHEMIX_JAX_PYTHON:-${ALLTHEMIX_JAX_ENV}/bin/python}"
    export ALLTHEMIX_XLA_PYTHON="${ALLTHEMIX_XLA_PYTHON:-${ALLTHEMIX_XLA_ENV}/bin/python}"

    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${state_root}/cache/pip}"
    export HF_HOME="${HF_HOME:-${state_root}/cache/huggingface}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
    export TORCH_HOME="${TORCH_HOME:-${state_root}/cache/torch}"
    export TMPDIR="${ALLTHEMIX_TMPDIR:-${state_root}/tmp}"
    export PYTHONNOUSERSITE=1

    mkdir -p \
        "${state_root}/venvs" \
        "${PIP_CACHE_DIR}" \
        "${HUGGINGFACE_HUB_CACHE}" \
        "${TORCH_HOME}" \
        "${TMPDIR}" \
        "${state_root}/xla"
}

allthemix_reset_backend_shell() {
    if declare -F deactivate >/dev/null 2>&1; then
        deactivate
    fi

    unalias python python3 pip pip3 2>/dev/null || true
    unset PYTHONHOME PYTHONPATH PIP_USER PIP_TARGET VIRTUAL_ENV
    unset PJRT_DEVICE XLA_FLAGS LIBTPU_INIT_ARGS JAX_PLATFORMS
    unset ALLTHEMIX_BACKEND PYTHON
    hash -r
}

allthemix_assert_python_prefix() {
    local python="$1"
    local expected_prefix="$2"

    if [[ ! -x "${python}" ]]; then
        echo "Virtual-environment Python is missing: ${python}" >&2
        return 1
    fi

    PYTHONNOUSERSITE=1 "${python}" - "${expected_prefix}" <<'PY'
from pathlib import Path
import sys

expected = Path(sys.argv[1]).expanduser().resolve()
actual = Path(sys.prefix).resolve()
if actual != expected:
    raise SystemExit(
        f"Malformed virtual environment: expected sys.prefix={expected}, "
        f"got {actual}; python={sys.executable}"
    )
PY
}

allthemix_resolve_python() {
    local python="$1"
    if [[ "${python}" == */* ]]; then
        local directory filename
        directory="$(cd "$(dirname "${python}")" && pwd)"
        filename="$(basename "${python}")"
        printf '%s/%s\n' "${directory}" "${filename}"
    else
        command -v "${python}"
    fi
}

allthemix_python_environment() {
    local python
    python="$(allthemix_resolve_python "$1")"
    cd "$(dirname "${python}")/.." && pwd
}

allthemix_export_backend_process_environment() {
    local backend="$1"
    local python="$2"
    local environment
    environment="$(allthemix_python_environment "${python}")"
    python="$(allthemix_resolve_python "${python}")"

    export VIRTUAL_ENV="${environment}"
    export PATH="$(dirname "${python}"):${PATH}"
    export PYTHON="${python}"
    export PYTHONNOUSERSITE=1
    unset PYTHONHOME PYTHONPATH PIP_USER PIP_TARGET

    if [[ "${backend}" == "xla" ]]; then
        export PJRT_DEVICE=TPU
    else
        unset PJRT_DEVICE XLA_FLAGS LIBTPU_INIT_ARGS JAX_PLATFORMS
    fi
}

allthemix_validate_backend_python() {
    local backend="$1"
    local python environment repo_root
    python="$(allthemix_resolve_python "$2")"
    environment="$(allthemix_python_environment "${python}")"
    repo_root="$(allthemix_environment_repo_root)"

    allthemix_assert_python_prefix "${python}" "${environment}"
    PYTHONNOUSERSITE=1 "${python}" \
        "${repo_root}/allthemix/utils/backend_environment.py" \
        --expected "${backend}" \
        --expected-prefix "${environment}" \
        --quiet
}
