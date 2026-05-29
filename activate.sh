#!/usr/bin/env bash
# Usage: source ./activate.sh
# Activates the local uv venv and pins all model/library caches under this
# workspace so nothing escapes to ~/.cache.

__COZ_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6 >/dev/null 2>&1 || true
fi

export HF_HOME="${__COZ_ROOT}/.hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${__COZ_ROOT}/.torch_cache"
export UV_CACHE_DIR="${__COZ_ROOT}/.cache/uv"
export PIP_CACHE_DIR="${__COZ_ROOT}/.cache/pip"
export HF_HUB_DISABLE_TELEMETRY=1

# shellcheck disable=SC1091
source "${__COZ_ROOT}/.venv/bin/activate"

export PYTHONPATH="${__COZ_ROOT}:${PYTHONPATH:-}"

unset __COZ_ROOT
