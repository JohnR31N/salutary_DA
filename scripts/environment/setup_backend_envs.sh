#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/backend_common.sh"

REPO_ROOT="$(allthemix_environment_repo_root)"
cd "${REPO_ROOT}"

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

if [[ -n "${PYTHON_BOOTSTRAP:-}" ]]; then
    BOOTSTRAP_PYTHON="${PYTHON_BOOTSTRAP}"
elif [[ -x /usr/bin/python3.10 ]]; then
    BOOTSTRAP_PYTHON=/usr/bin/python3.10
elif command -v python3.10 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="$(command -v python3.10)"
else
    BOOTSTRAP_PYTHON="$(command -v python3)"
fi

RECREATE="${RECREATE:-false}"
RUNTIME_CHECK="${RUNTIME_CHECK:-true}"
BOOTSTRAP_TOOLS_DIR="${ALLTHEMIX_STATE_ROOT}/bootstrap"

move_environment_aside() {
    local environment="$1"
    local reason="$2"
    local backup
    backup="${environment}.${reason}.$(date +%Y%m%d_%H%M%S)"
    echo "Moving ${environment} to ${backup}"
    mv "${environment}" "${backup}"
}

create_with_virtualenv() {
    local environment="$1"
    local virtualenv_root="${BOOTSTRAP_TOOLS_DIR}/virtualenv"

    mkdir -p "${virtualenv_root}"
    if ! env \
        -u PYTHONHOME -u PYTHONPATH -u PYTHONNOUSERSITE \
        PYTHONPATH="${virtualenv_root}" \
        "${BOOTSTRAP_PYTHON}" -c "import virtualenv" \
        >/dev/null 2>&1; then
        echo "Bootstrapping virtualenv without sudo into ${virtualenv_root}"
        if ! env \
            -u PYTHONHOME -u PYTHONPATH -u PYTHONNOUSERSITE \
            -u PIP_USER -u PIP_TARGET -u PIP_PREFIX \
            -u PIP_REQUIRE_VIRTUALENV \
            "${BOOTSTRAP_PYTHON}" -m pip --version >/dev/null 2>&1; then
            echo "${BOOTSTRAP_PYTHON} has neither ensurepip nor pip." >&2
            echo "Set PYTHON_BOOTSTRAP to a Python 3.10 interpreter with pip." >&2
            return 1
        fi
        env \
            -u PYTHONHOME -u PYTHONPATH -u PYTHONNOUSERSITE \
            -u PIP_USER -u PIP_TARGET -u PIP_PREFIX \
            -u PIP_REQUIRE_VIRTUALENV \
            PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
            "${BOOTSTRAP_PYTHON}" -m pip install \
            --upgrade \
            --target "${virtualenv_root}" \
            "virtualenv>=20.26,<21"
    fi

    env \
        -u PYTHONHOME -u PYTHONNOUSERSITE \
        PYTHONNOUSERSITE=1 \
        PYTHONPATH="${virtualenv_root}" \
        "${BOOTSTRAP_PYTHON}" -m virtualenv \
        --copies \
        "${environment}"
}

create_environment() {
    local environment="$1"
    local python="${environment}/bin/python"

    echo "Creating environment with stdlib venv: ${environment}"
    if "${BOOTSTRAP_PYTHON}" -m venv --copies "${environment}" && \
        PYTHONNOUSERSITE=1 "${python}" -m pip --version \
        >/dev/null 2>&1; then
        return
    fi

    echo "stdlib venv could not seed pip; falling back to virtualenv."
    if [[ -e "${environment}" ]]; then
        move_environment_aside "${environment}" incomplete
    fi
    create_with_virtualenv "${environment}"
}

install_backend() {
    local backend="$1"
    local environment python requirements
    environment="$(allthemix_backend_env "${backend}")"
    python="${environment}/bin/python"
    requirements="${REPO_ROOT}/requirements-${backend}.txt"

    if [[ -e "${environment}" && "${RECREATE}" == "true" ]]; then
        move_environment_aside "${environment}" old
    fi

    if [[ ! -d "${environment}" ]]; then
        create_environment "${environment}"
    elif [[ ! -x "${python}" ]] || ! \
        PYTHONNOUSERSITE=1 "${python}" -m pip --version \
        >/dev/null 2>&1; then
        echo "Existing ${backend} environment is incomplete (pip is missing)."
        move_environment_aside "${environment}" incomplete
        create_with_virtualenv "${environment}"
    fi

    if ! allthemix_assert_python_prefix "${python}" "${environment}"; then
        echo "Set RECREATE=true and rerun this command to replace it safely." >&2
        return 1
    fi
    if grep -Eiq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' \
        "${environment}/pyvenv.cfg"; then
        echo "System site packages are enabled in ${environment}." >&2
        echo "Set RECREATE=true and rerun this command." >&2
        return 1
    fi

    if ! PYTHONNOUSERSITE=1 "${python}" - "${backend}" <<'PY'
import importlib.metadata
import sys

backend = sys.argv[1]
forbidden = ("torch-xla",) if backend == "jax" else ("jax", "jaxlib")
installed = []
for name in forbidden:
    try:
        installed.append(f"{name}={importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
        pass
if installed:
    raise SystemExit(
        f"The existing {backend} environment contains the other TPU "
        f"backend: {', '.join(installed)}"
    )
PY
    then
        echo "Set RECREATE=true and rerun this command." >&2
        return 1
    fi

    echo "Installing requirements-${backend}.txt with ${python}"
    env \
        -u PYTHONHOME -u PYTHONPATH -u PIP_USER -u PIP_TARGET \
        PYTHONNOUSERSITE=1 \
        PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
        "${python}" -m pip install --upgrade pip wheel
    env \
        -u PYTHONHOME -u PYTHONPATH -u PIP_USER -u PIP_TARGET \
        PYTHONNOUSERSITE=1 \
        PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
        "${python}" -m pip install -r "${requirements}"
    PYTHONNOUSERSITE=1 "${python}" -m pip check

    local runtime_args=()
    if [[ "${RUNTIME_CHECK}" == "true" ]]; then
        runtime_args=(--runtime --require-tpu)
    fi

    if [[ "${backend}" == "xla" ]]; then
        env \
            -u PYTHONHOME -u PYTHONPATH -u PIP_USER -u PIP_TARGET \
            -u XLA_FLAGS -u LIBTPU_INIT_ARGS -u JAX_PLATFORMS \
            PYTHONNOUSERSITE=1 PJRT_DEVICE=TPU \
            "${python}" "${REPO_ROOT}/allthemix/utils/backend_environment.py" \
            --expected xla \
            --expected-prefix "${environment}" \
            "${runtime_args[@]}"
    else
        env \
            -u PYTHONHOME -u PYTHONPATH -u PIP_USER -u PIP_TARGET \
            -u PJRT_DEVICE -u XLA_FLAGS -u LIBTPU_INIT_ARGS \
            -u JAX_PLATFORMS \
            PYTHONNOUSERSITE=1 \
            "${python}" "${REPO_ROOT}/allthemix/utils/backend_environment.py" \
            --expected jax \
            --expected-prefix "${environment}" \
            "${runtime_args[@]}"
    fi

    echo "${backend} environment is ready: ${environment}"
}

echo "Repository: ${REPO_ROOT}"
echo "State root: ${ALLTHEMIX_STATE_ROOT}"
echo "Bootstrap Python: ${BOOTSTRAP_PYTHON}"
echo "Runtime check: ${RUNTIME_CHECK}"

for backend in "${BACKENDS[@]}"; do
    install_backend "${backend}"
done

echo
echo "Switch environments with:"
echo "  source scripts/environment/use_backend.sh xla"
echo "  source scripts/environment/use_backend.sh jax"
