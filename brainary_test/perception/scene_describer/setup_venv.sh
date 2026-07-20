#!/usr/bin/env bash
# Build the isolated venv for the VLM describe server (keeps openai/key out of the isaac env).
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VLM_VENV:-.venv_vlm}"
PY="${PYTHON:-python3}"

if [ ! -d "$VENV" ]; then
    echo "[setup] creating $VENV with $PY"
    "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt
echo "[setup] done. run:"
echo "  OPENAI_API_KEY=sk-... $VENV/bin/python vlm_describe_server.py"
