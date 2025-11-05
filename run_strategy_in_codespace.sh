#!/bin/bash
# 在 GitHub Codespace 中运行策略优化脚本
# 使用方法：在 Cursor 的远程终端中运行：bash run_strategy_in_codespace.sh

echo "🚀 开始运行策略优化..."
echo "=========================================="

# 检查是否在 Codespace 环境中
if [ -z "$CODESPACE_NAME" ] && [[ ! "$(hostname)" =~ codespaces ]]; then
    echo "⚠️  警告：可能不在 Codespace 环境中"
    echo "当前主机名: $(hostname)"
    echo "当前目录: $(pwd)"
else
    echo "✅ 确认在 Codespace 环境中"
    echo "当前目录: $(pwd)"
fi

echo ""
echo "步骤 1: 检查文件..."
if [ ! -f "smart_robust_strategy_v2.py" ]; then
    echo "❌ 错误: smart_robust_strategy_v2.py 不存在"
    exit 1
fi

if [ ! -f "rb0_data.csv" ]; then
    echo "❌ 错误: rb0_data.csv 不存在"
    exit 1
fi

if [ ! -f "colab_complete_script.py" ]; then
    echo "❌ 错误: colab_complete_script.py 不存在"
    exit 1
fi

echo "✅ 所有必需文件存在"

echo ""
echo "步骤 2: 检查 Python 环境..."
python --version
python3 --version

echo ""
echo "步骤 3: 安装依赖..."
pip install -q pandas numpy
echo "✅ 依赖安装完成"

echo ""
echo "步骤 4: 运行策略优化脚本..."
echo "=========================================="
echo "⚠️  这可能需要 15-30 分钟，请耐心等待..."
echo "=========================================="
echo ""

python colab_complete_script.py

echo ""
echo "=========================================="
echo "✅ 运行完成！"
echo "=========================================="
echo ""
echo "结果文件："
if [ -f "strategy_optimization_results.csv" ]; then
    echo "  ✅ strategy_optimization_results.csv"
    echo "     行数: $(wc -l < strategy_optimization_results.csv)"
fi

if [ -f "best_strategy_config.json" ]; then
    echo "  ✅ best_strategy_config.json"
    echo "     内容:"
    cat best_strategy_config.json | head -20
fi

echo ""
echo "💡 提示：可以在 Cursor 的文件浏览器中查看结果文件"

