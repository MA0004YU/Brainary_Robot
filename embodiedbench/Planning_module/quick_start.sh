#!/bin/bash
# 一键启动 Intent Agent 交互测试（需已启动 EmbodiedLTM）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate embench

INSTRUCTION="${1:-Get me a can of coke.}"

echo "========================================="
echo "   启动 Intent Reasoning Agent 测试"
echo "========================================="
echo "场景: EB-Habitat base 评测集 (episode index 3)"
echo "指令: ${INSTRUCTION}"
echo ""

xvfb-run -a python interactive_test.py --instruction "${INSTRUCTION}"
