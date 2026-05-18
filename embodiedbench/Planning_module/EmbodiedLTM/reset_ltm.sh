#!/bin/bash
# 清空并重置 LTM 图谱与本地缓存，重启 uvicorn 服务。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LTM_ROOT="$SCRIPT_DIR"
MEMORY_KB="$LTM_ROOT/MemoryKB"

echo "[ResetLTM] 1. 停止已有 uvicorn 服务..."
pkill -f "uvicorn app:app" || true
sleep 1

echo "[ResetLTM] 2. 清空 GraphRAG 工作目录..."
rm -rf "$MEMORY_KB/Long_Term_Memory/Graph_Construction/MMKG"
rm -rf "$MEMORY_KB/Long_Term_Memory/memory_chunks"
rm -rf "$MEMORY_KB/User_Conversation"

mkdir -p "$MEMORY_KB/Long_Term_Memory/Graph_Construction/MMKG/episodic"
mkdir -p "$MEMORY_KB/Long_Term_Memory/Graph_Construction/MMKG/semantic"
mkdir -p "$MEMORY_KB/Long_Term_Memory/memory_chunks"
mkdir -p "$MEMORY_KB/User_Conversation"

echo "[ResetLTM] 3. 启动 LTM 服务 (port 8000)..."
cd "$LTM_ROOT"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate embodiedltm 2>/dev/null || true
fi
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

echo "[ResetLTM] 4. 等待服务就绪..."
sleep 3
echo "[ResetLTM] 完成。LTM 已重置并重新上线。"
