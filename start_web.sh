#!/bin/bash

echo "=================================="
echo "🚀 启动螺纹钢RSI Web监控系统"
echo "=================================="
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null
then
    echo "❌ 错误: 未找到Python3"
    echo "请先安装Python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Flask未安装，正在安装..."
    pip3 install flask pandas numpy akshare
fi

echo "✓ 依赖检查完成"
echo ""

# 使用5001端口（避免冲突）
PORT=5001

# 获取本机IP
echo "📍 本机访问地址:"
echo "   http://localhost:$PORT"
echo ""
echo "📱 局域网访问地址:"
IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
if [ ! -z "$IP" ]; then
    echo "   http://$IP:$PORT"
else
    echo "   无法获取IP地址"
fi
echo ""
echo "💡 注意: 使用5001端口（如被占用会自动尝试其他端口）"
echo ""

# 启动服务器
echo "🌐 启动Web服务器..."
echo "按 Ctrl+C 停止服务器"
echo "=================================="
echo ""

python3 web_rsi_monitor.py

