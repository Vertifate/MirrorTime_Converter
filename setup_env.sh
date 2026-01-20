#!/bin/bash
# MirrorTime Converter 环境设置脚本
#WDD [2026-01-19] [环境配置脚本]

echo "🔧 设置 MirrorTime Converter 开发环境..."
echo ""

# 激活 conda 环境
echo "📦 激活 conda 环境: mirrortime"
eval "$(conda shell.bash hook)"
conda activate mirrortime

# 安装 Python 依赖
echo ""
echo "📥 安装 Python 依赖..."
pip install -r requirements.txt

# 安装前端依赖
echo ""
echo "📥 安装前端依赖..."
cd visualization-ui
npm install
cd ..

echo ""
echo "✅ 环境设置完成！"
echo ""
echo "下一步："
echo "  1. 运行: conda activate mirrortime"
echo "  2. 启动: python launch.py"
echo ""
