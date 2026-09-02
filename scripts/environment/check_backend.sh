#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/backend_common.sh"

REPO_ROOT="$(allthemix_environment_repo_root)"

REQUESTED_BACKEND="${1:-all}"
case "${REQUESTED_BACKEND}" in
    jax) BACKENDS=(jax) ;;
    xla) BACKENDS=(xla) ;;
    all) BACKENDS=(jax xla) ;;
    *)
        echo "Usage: bash $0 [jax|xla|all]" >&2
        exit 2
        ;;
esac
allthemix_configure_storage

for backend in "${BACKENDS[@]}"; do
    environment="$(allthemix_backend_env "${backend}")"
    python="${environment}/bin/python"
    echo "============================================================"
    echo "Checking ${backend}: ${python}"
    allthemix_assert_python_prefix "${python}" "${environment}"
    PYTHONNOUSERSITE=1 "${python}" -m pip check

    if [[ "${backend}" == "xla" ]]; then
        env \
            -u PYTHONHOME -u PYTHONPATH -u PIP_USER -u PIP_TARGET \
            -u XLA_FLAGS -u LIBTPU_INIT_ARGS -u JAX_PLATFORMS \
            PYTHONNOUSERSITE=1 PJRT_DEVICE=TPU \
            "${python}" "${REPO_ROOT}/allthemix/utils/backend_environment.py" \
            --expected xla \
            --expected-prefix "${environment}" \
            --runtime \
            --require-tpu
    else
        env \
            -u PYTHONHOME -u PYTHONPATH -u PIP_USER -u PIP_TARGET \
            -u PJRT_DEVICE -u XLA_FLAGS -u LIBTPU_INIT_ARGS \
            -u JAX_PLATFORMS \
            PYTHONNOUSERSITE=1 \
            "${python}" "${REPO_ROOT}/allthemix/utils/backend_environment.py" \
            --expected jax \
            --expected-prefix "${environment}" \
            --runtime \
            --require-tpu
    fi
done
