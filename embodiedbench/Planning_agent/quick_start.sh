#!/bin/bash
# 激活 conda 环境 (忽略执行报错以便向下兼容)
source ~/anaconda3/etc/profile.d/conda.sh || true
conda activate embench

echo "========================================="
echo "   启动 Intent Reasoning Agent 测试场景   "
echo "========================================="
echo "场景: Episode 3 (解渴替代意图验证)"
echo "指令: I am thirsty, bring me the apple from the TV stand to the sink."
echo ""

# 使用 xvfb-run 以 Headless 模式运行图形仿真引擎
xvfb-run -a python interactive_test.py --new_session --instruction "I am thirsty, bring me the apple from the TV stand to the sink."
