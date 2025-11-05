# 策略优化项目 - GitHub Codespaces 使用指南

## 🚀 快速开始

### 在 GitHub Codespaces 中运行

1. **创建 Codespace**
   - 在 GitHub 仓库页面点击绿色的 "Code" 按钮
   - 选择 "Codespaces" 标签
   - 点击 "Create codespace on main"

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行优化脚本**
   ```bash
   python colab_complete_script.py
   ```

4. **查看结果**
   - `strategy_optimization_results.csv` - 所有参数组合的结果
   - `best_strategy_config.json` - 最优配置

### 在 Cursor 中连接 Codespace

1. 在 Codespace 页面点击右上角 "..." → "Open in Visual Studio Code"
2. Cursor 会自动打开并连接到 Codespace
3. 或者使用命令面板：`Cmd/Ctrl + Shift + P` → "Codespaces: Connect to Codespace"

## 📁 文件说明

- `smart_robust_strategy_v2.py` - 策略主文件
- `rb0_data.csv` - 历史数据（7.22 MB）
- `colab_complete_script.py` - 完整优化脚本（推荐使用）
- `colab_optimization.py` - 简化优化脚本
- `delete/jqdata_fetch_and_analyze.py` - 依赖的工具函数

## ⚙️ 优化参数

默认参数范围：
- 止盈：4-9 点
- 止损：5-11 点
- 评分阈值：6.5-8.0
- 市场过滤：True/False

可以在脚本中修改这些参数范围。

## 💡 提示

- Codespace 免费版：2 核 CPU，4GB 内存
- 运行时间：约 15-30 分钟（取决于参数组合数）
- 30 分钟无活动后会自动停止
- 记得下载结果文件或提交到 Git

