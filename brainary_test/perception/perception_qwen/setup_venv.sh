#!/usr/bin/env bash
# Build the venv for the Qwen2.5-VL server. Uses --system-site-packages so it INHERITS the base
# conda env's working CUDA torch (no multi-GB torch reinstall); only adds transformers/qwen-vl-utils.
set -euo pipefail
cd "$(dirname "$0")"

VENV="${QWEN_VENV:-.venv_qwen}"
# base conda python (has torch + CUDA). Override with PYTHON=... if your torch lives elsewhere.
PY="${PYTHON:-/home1/banghai/miniconda3/bin/python}"

echo "[setup] base python = $PY"
"$PY" -c "import torch; print('[setup] inherited torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ ! -d "$VENV" ]; then
    echo "[setup] creating $VENV (--system-site-packages, inherits torch)"
    "$PY" -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt
"$VENV/bin/python" -c "import transformers, qwen_vl_utils; print('[setup] transformers', transformers.__version__)"
echo "[setup] done. run (first start downloads ~7GB Qwen weights):"
echo "  $VENV/bin/python qwen_perception_server.py"
echo "  (slow download? export HF_ENDPOINT=https://hf-mirror.com first)"
