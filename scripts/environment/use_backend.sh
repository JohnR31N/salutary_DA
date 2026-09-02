#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced so it can update the current shell:" >&2
    echo "  source scripts/environment/use_backend.sh <jax|xla>" >&2
    exit 2
fi

_allthemix_use_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_allthemix_use_script_dir}/backend_common.sh"

_allthemix_backend="${1:-}"
_allthemix_runtime="${2:-}"
case "${_allthemix_backend}" in
    jax|xla) ;;
    *)
        echo "Usage: source scripts/environment/use_backend.sh <jax|xla> [--runtime]" >&2
        return 2
        ;;
esac
if [[ -n "${_allthemix_runtime}" && "${_allthemix_runtime}" != "--runtime" ]]; then
    echo "Unknown option: ${_allthemix_runtime}" >&2
    return 2
fi

_allthemix_repo_root="$(allthemix_environment_repo_root)"
_allthemix_environment="$(allthemix_backend_env "${_allthemix_backend}")"
_allthemix_python="${_allthemix_environment}/bin/python"

allthemix_reset_backend_shell
allthemix_configure_storage

if ! allthemix_assert_python_prefix \
    "${_allthemix_python}" "${_allthemix_environment}"; then
    echo "Create or repair it with:" >&2
    echo "  bash scripts/environment/setup_backend_envs.sh ${_allthemix_backend}" >&2
    return 1
fi

# shellcheck disable=SC1090
source "${_allthemix_environment}/bin/activate"
export ALLTHEMIX_BACKEND="${_allthemix_backend}"
export PYTHON="${_allthemix_python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH PIP_USER PIP_TARGET

_allthemix_check_args=(
    --expected "${_allthemix_backend}"
    --expected-prefix "${_allthemix_environment}"
    --quiet
)
if [[ "${_allthemix_runtime}" == "--runtime" ]]; then
    _allthemix_check_args+=(--runtime --require-tpu)
fi

if [[ "${_allthemix_backend}" == "xla" ]]; then
    export PJRT_DEVICE=TPU
    "${_allthemix_python}" \
        "${_allthemix_repo_root}/allthemix/utils/backend_environment.py" \
        "${_allthemix_check_args[@]}" || return 1
else
    unset PJRT_DEVICE XLA_FLAGS LIBTPU_INIT_ARGS JAX_PLATFORMS
    "${_allthemix_python}" \
        "${_allthemix_repo_root}/allthemix/utils/backend_environment.py" \
        "${_allthemix_check_args[@]}" || return 1
fi

hash -r
echo "AllTheMix backend: ${ALLTHEMIX_BACKEND}"
echo "Python: $(command -v python)"
echo "Prefix: ${VIRTUAL_ENV}"
echo "User site disabled: ${PYTHONNOUSERSITE}"

unset \
    _allthemix_backend \
    _allthemix_check_args \
    _allthemix_environment \
    _allthemix_python \
    _allthemix_repo_root \
    _allthemix_runtime \
    _allthemix_use_script_dir
