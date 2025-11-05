"""
策略参数优化 - Google Colab 完整脚本
可以直接复制到Colab中运行
"""

# ============================================================================
# 第一部分：安装依赖和上传文件
# ============================================================================

print("=" * 80)
print("步骤1：安装依赖")
print("=" * 80)

# 安装依赖
import subprocess
import sys

try:
    import pandas
    import numpy
    print("✅ pandas 和 numpy 已安装")
except ImportError:
    print("📦 正在安装 pandas 和 numpy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy"])
    print("✅ 安装完成")

# ============================================================================
# 第二部分：上传文件（如果需要）
# ============================================================================

print("\n" + "=" * 80)
print("步骤2：检查文件")
print("=" * 80)

import os

# 检查文件是否存在（支持大小写不敏感）
files_needed = {
    'rb0_data.csv': ['rb0_data.csv', 'RB0_data.csv', 'RB0_data (1).csv'],  # 支持多种文件名
    'smart_robust_strategy_v2.py': ['smart_robust_strategy_v2.py']
}

files_found = {}
actual_filenames = {}

for key, possible_names in files_needed.items():
    found = False
    for filename in possible_names:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024 / 1024
            print(f"✅ {filename} 已存在 ({size:.2f} MB)")
            files_found[key] = True
            actual_filenames[key] = filename
            found = True
            break
    
    if not found:
        # 尝试大小写不敏感搜索
        all_files = os.listdir('.')
        for f in all_files:
            if key.lower() in f.lower() and (key.endswith('.csv') and f.endswith('.csv') or 
                                             key.endswith('.py') and f.endswith('.py')):
                size = os.path.getsize(f) / 1024 / 1024
                print(f"✅ 找到类似文件: {f} ({size:.2f} MB)")
                files_found[key] = True
                actual_filenames[key] = f
                found = True
                break
    
    if not found:
        print(f"❌ {key} 不存在")
        files_found[key] = False

# 如果文件不存在，提示上传
if not all(files_found.values()):
    print("\n⚠️  需要上传缺失的文件")
    
    # 尝试自动上传（如果是在Colab环境中）
    try:
        from google.colab import files
        
        for key, possible_names in files_needed.items():
            if not files_found[key]:
                print(f"\n📥 请上传 {key} 文件")
                uploaded = files.upload()
                # 检查上传的文件
                for uploaded_name in uploaded.keys():
                    if key.lower() in uploaded_name.lower():
                        actual_filenames[key] = uploaded_name
                        files_found[key] = True
                        print(f"✅ {uploaded_name} 上传成功")
                        break
    except ImportError:
        print("⚠️  不在Colab环境中，请手动上传文件")

# 检查所有文件是否就绪
if not all(files_found.values()):
    print("\n❌ 文件不完整，请先上传所有必需文件")
    print("\n💡 提示：如果文件已上传但文件名不同，请检查：")
    print("  1. 文件名是否正确（支持大小写变化）")
    print("  2. 文件是否在正确的目录")
    print("\n当前目录的文件：")
    for f in os.listdir('.'):
        if '.csv' in f or 'smart_robust' in f:
            print(f"  - {f}")
    raise FileNotFoundError("缺少必需文件")

print("\n✅ 所有文件已就绪！")

# ============================================================================
# 第三部分：加载数据和导入策略
# ============================================================================

print("\n" + "=" * 80)
print("步骤3：加载数据")
print("=" * 80)

import pandas as pd
import numpy as np

# 加载数据（使用实际文件名）
data_file = actual_filenames.get('rb0_data.csv', 'rb0_data.csv')
df = pd.read_csv(data_file, parse_dates=['date', 'datetime'])
print(f"✅ 加载 {len(df)} 条数据")
print(f"   日期范围: {df['date'].min()} 至 {df['date'].max()}")

print("\n" + "=" * 80)
print("步骤4：导入策略模块")
print("=" * 80)

import importlib.util

# 导入策略模块
spec = importlib.util.spec_from_file_location(
    "smart_robust_strategy_v2",
    "smart_robust_strategy_v2.py"
)
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)
smart_robust_strategy_single_day = strategy_module.smart_robust_strategy_single_day
print("✅ 策略模块导入成功")

# ============================================================================
# 第四部分：优化函数
# ============================================================================

def test_strategy_no_parallel(df, take_profit, stop_loss, min_score, use_market_filter):
    """测试策略（不使用并行）"""
    if df is None or len(df) == 0:
        return []
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    all_dates = sorted(df['date'].unique())
    
    all_trades = []
    for date in all_dates:
        df_day = df[df['date'] == date].reset_index(drop=True)
        if len(df_day) >= 50:
            trades = smart_robust_strategy_single_day(
                df_day, date,
                take_profit=take_profit,
                stop_loss=stop_loss,
                max_holding_bars=30,
                min_score=min_score,
                use_market_filter=use_market_filter,
                avoid_bad_periods=['2024-11', '2024-12'],
                use_entry_timing_optimization=False
            )
            all_trades.extend(trades)
    
    return all_trades

def evaluate_params(params, df):
    """评估单个参数组合"""
    take_profit, stop_loss, min_score, use_market_filter = params
    
    try:
        trades = test_strategy_no_parallel(df, take_profit, stop_loss, min_score, use_market_filter)
        
        if len(trades) == 0:
            return {
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'min_score': min_score,
                'use_market_filter': use_market_filter,
                'trade_count': 0,
                'win_rate': 0,
                'total_profit': 0,
                'theoretical_total_profit': 0,
                'avg_profit': 0,
                'avg_theoretical_profit': 0,
                'profit_per_trade': 0,
                'loss_count': 0,
                'win_count': 0,
                'score': -9999
            }
        
        df_trades = pd.DataFrame(trades)
        df_trades['date'] = pd.to_datetime(df_trades['date'])
        
        total_profit = df_trades['profit'].sum()
        win_rate = len(df_trades[df_trades['profit'] > 0]) / len(df_trades) * 100
        avg_profit = df_trades['profit'].mean()
        
        theoretical_profits = df_trades.get('theoretical_profit', df_trades['profit'])
        theoretical_total_profit = theoretical_profits.sum()
        avg_theoretical_profit = theoretical_profits.mean()
        
        win_count = len(df_trades[df_trades['profit'] > 0])
        loss_count = len(df_trades[df_trades['profit'] <= 0])
        
        winning_trades = df_trades[df_trades['profit'] > 0]
        losing_trades = df_trades[df_trades['profit'] <= 0]
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            avg_win = winning_trades['theoretical_profit'].mean() if 'theoretical_profit' in winning_trades.columns else winning_trades['profit'].mean()
            avg_loss = abs(losing_trades['theoretical_profit'].mean()) if 'theoretical_profit' in losing_trades.columns else abs(losing_trades['profit'].mean())
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_loss_ratio = 0
        
        trade_count_penalty = min(1.0, len(df_trades) / 100)
        balance_score = (
            theoretical_total_profit * 0.5 +
            win_rate * 0.3 +
            (1 - trade_count_penalty) * 50 * 0.2
        )
        
        return {
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'min_score': min_score,
            'use_market_filter': use_market_filter,
            'trade_count': len(df_trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'theoretical_total_profit': theoretical_total_profit,
            'avg_profit': avg_profit,
            'avg_theoretical_profit': avg_theoretical_profit,
            'profit_per_trade': theoretical_total_profit / len(df_trades) if len(df_trades) > 0 else 0,
            'loss_count': loss_count,
            'win_count': win_count,
            'profit_loss_ratio': profit_loss_ratio,
            'balance_score': balance_score,
            'core_score': theoretical_total_profit
        }
    except Exception as e:
        print(f"Error evaluating params {params}: {e}")
        return None

# ============================================================================
# 第五部分：执行优化
# ============================================================================

print("\n" + "=" * 80)
print("步骤5：开始优化")
print("=" * 80)

from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import json

# 参数搜索空间
take_profit_range = [4, 5, 6, 7, 8, 9]
stop_loss_range = [5, 6, 7, 8, 9, 10, 11]
min_score_range = [6.5, 7.0, 7.5, 8.0]
use_market_filter_range = [True, False]

param_combinations = list(product(
    take_profit_range,
    stop_loss_range,
    min_score_range,
    use_market_filter_range
))

# 过滤：止损必须大于等于止盈
param_combinations = [p for p in param_combinations if p[1] >= p[0]]

print(f"总参数组合数: {len(param_combinations)}")
print(f"测试参数范围:")
print(f"  止盈: {min(take_profit_range)}-{max(take_profit_range)}点")
print(f"  止损: {min(stop_loss_range)}-{max(stop_loss_range)}点")
print(f"  评分阈值: {min_score_range}")
print(f"  市场过滤: {use_market_filter_range}")

max_workers = max(1, cpu_count() - 1)
print(f"\n使用 {max_workers} 个进程并行优化...")
print(f"CPU核心数: {cpu_count()}")

start_time = time.time()

# 分批处理
batch_size = max_workers * 2
results = []
total_batches = (len(param_combinations) + batch_size - 1) // batch_size

for batch_idx in range(total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(param_combinations))
    batch_params = param_combinations[start_idx:end_idx]
    
    progress = (batch_idx + 1) / total_batches * 100
    print(f"处理批次 {batch_idx + 1}/{total_batches} ({len(batch_params)} 个参数组合) - 进度: {progress:.1f}%", end='\r')
    
    with Pool(processes=max_workers) as pool:
        evaluate_func = partial(evaluate_params, df=df)
        batch_results = pool.map(evaluate_func, batch_params)
    
    batch_results = [r for r in batch_results if r is not None]
    results.extend(batch_results)

elapsed_time = time.time() - start_time
print(f"\n✅ 优化完成，耗时 {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")
print(f"有效参数组合: {len(results)}/{len(param_combinations)}")

# ============================================================================
# 第六部分：分析结果
# ============================================================================

print("\n" + "=" * 80)
print("步骤6：分析结果")
print("=" * 80)

df_results = pd.DataFrame(results)

# TOP 10 按理论总收益排序
print("\n【TOP 10】按理论总收益排序:")
print("-" * 80)
top_by_theoretical = df_results.nlargest(10, 'theoretical_total_profit')
for idx, row in top_by_theoretical.iterrows():
    print(f"止盈{row['take_profit']}点 / 止损{row['stop_loss']}点 / 评分{row['min_score']:.1f} / 市场过滤{row['use_market_filter']}")
    print(f"  理论总收益: {row['theoretical_total_profit']:+.2f}点")
    print(f"  实际总收益: {row['total_profit']:+.2f}点")
    print(f"  交易次数: {row['trade_count']}笔")
    print(f"  胜率: {row['win_rate']:.2f}%")
    print(f"  平均每笔理论收益: {row['avg_theoretical_profit']:+.2f}点")
    print(f"  盈亏比: {row['profit_loss_ratio']:.2f}")
    print()

# 推荐最优组合
filtered_results = df_results[
    (df_results['win_rate'] >= 65) &
    (df_results['trade_count'] >= 20) &
    (df_results['theoretical_total_profit'] > 0)
]

if len(filtered_results) > 0:
    best = filtered_results.nlargest(1, 'theoretical_total_profit').iloc[0]
    print("\n" + "=" * 80)
    print("🎯 推荐最优组合")
    print("=" * 80)
    print(f"止盈: {best['take_profit']}点")
    print(f"止损: {best['stop_loss']}点")
    print(f"评分阈值: {best['min_score']:.1f}分")
    print(f"市场过滤: {best['use_market_filter']}")
    print(f"\n表现指标:")
    print(f"  理论总收益: {best['theoretical_total_profit']:+.2f}点")
    print(f"  实际总收益: {best['total_profit']:+.2f}点")
    print(f"  交易次数: {best['trade_count']}笔")
    print(f"  胜率: {best['win_rate']:.2f}%")
    print(f"  平均每笔理论收益: {best['avg_theoretical_profit']:+.2f}点")
    print(f"  盈亏比: {best['profit_loss_ratio']:.2f}")
    print(f"  平衡评分: {best['balance_score']:.2f}")
    
    best_config = {
        'take_profit': int(best['take_profit']),
        'stop_loss': int(best['stop_loss']),
        'min_score': float(best['min_score']),
        'use_market_filter': bool(best['use_market_filter']),
        'theoretical_total_profit': float(best['theoretical_total_profit']),
        'win_rate': float(best['win_rate']),
        'trade_count': int(best['trade_count'])
    }
    
    with open('best_strategy_config.json', 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 最优配置已保存到: best_strategy_config.json")
else:
    print("\n⚠️  没有找到满足约束条件的最优组合")

# 保存所有结果
df_results.to_csv('strategy_optimization_results.csv', index=False, encoding='utf-8-sig')
print(f"✅ 优化结果已保存到: strategy_optimization_results.csv")

# ============================================================================
# 第七部分：下载结果（Colab专用）
# ============================================================================

print("\n" + "=" * 80)
print("步骤7：下载结果")
print("=" * 80)

try:
    from google.colab import files
    
    print("📥 准备下载结果文件...")
    
    if os.path.exists('strategy_optimization_results.csv'):
        files.download('strategy_optimization_results.csv')
        print("✅ 已下载: strategy_optimization_results.csv")
    
    if os.path.exists('best_strategy_config.json'):
        files.download('best_strategy_config.json')
        print("✅ 已下载: best_strategy_config.json")
    
    print("\n🎉 所有结果已下载完成！")
except ImportError:
    print("⚠️  不在Colab环境中，结果文件已保存到当前目录：")
    print("  - strategy_optimization_results.csv")
    print("  - best_strategy_config.json")

print("\n" + "=" * 80)
print("✅ 优化流程全部完成！")
print("=" * 80)

