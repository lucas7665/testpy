"""
Google Colab 优化脚本
可以直接在Colab中运行
"""

# ============================================================================
# 步骤1：安装依赖（在Colab中运行）
# ============================================================================
# !pip install -q pandas numpy

# ============================================================================
# 步骤2：上传文件（在Colab中运行）
# ============================================================================
# from google.colab import files
# print("请上传 rb0_data.csv 文件")
# files.upload()
# print("请上传 smart_robust_strategy_v2.py 文件")
# files.upload()

# ============================================================================
# 步骤3：执行以下代码
# ============================================================================

import sys
import os
import pandas as pd
import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import json
import importlib.util

# 检查文件
if not os.path.exists('rb0_data.csv'):
    print("❌ rb0_data.csv 不存在，请先上传文件")
    sys.exit(1)

if not os.path.exists('smart_robust_strategy_v2.py'):
    print("❌ smart_robust_strategy_v2.py 不存在，请先上传文件")
    sys.exit(1)

# 加载数据
print("📥 加载数据...")
df = pd.read_csv('rb0_data.csv', parse_dates=['date', 'datetime'])
print(f"✅ 加载 {len(df)} 条数据")
print(f"   日期范围: {df['date'].min()} 至 {df['date'].max()}")

# 导入策略模块
print("\n📥 导入策略模块...")
spec = importlib.util.spec_from_file_location(
    "smart_robust_strategy_v2",
    "smart_robust_strategy_v2.py"
)
strategy_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_module)
smart_robust_strategy_single_day = strategy_module.smart_robust_strategy_single_day
print("✅ 策略模块导入成功")

# ============================================================================
# 优化函数
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
# 主优化流程
# ============================================================================

print("\n" + "=" * 120)
print("基于理论收益的策略优化")
print("=" * 120)

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

param_combinations = [p for p in param_combinations if p[1] >= p[0]]

print(f"\n总参数组合数: {len(param_combinations)}")
print(f"测试参数范围:")
print(f"  止盈: {min(take_profit_range)}-{max(take_profit_range)}点")
print(f"  止损: {min(stop_loss_range)}-{max(stop_loss_range)}点")
print(f"  评分阈值: {min_score_range}")
print(f"  市场过滤: {use_market_filter_range}")

max_workers = max(1, cpu_count() - 1)
print(f"\n使用 {max_workers} 个进程并行优化...")
print(f"CPU核心数: {cpu_count()}")
start_time = time.time()

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

df_results = pd.DataFrame(results)

print("\n" + "=" * 120)
print("最优参数组合（按理论总收益排序）")
print("=" * 120)

top_by_theoretical = df_results.nlargest(10, 'theoretical_total_profit')
print("\n【TOP 10】按理论总收益排序:")
print("-" * 120)
for idx, row in top_by_theoretical.iterrows():
    print(f"止盈{row['take_profit']}点 / 止损{row['stop_loss']}点 / 评分{row['min_score']:.1f} / 市场过滤{row['use_market_filter']}")
    print(f"  理论总收益: {row['theoretical_total_profit']:+.2f}点")
    print(f"  实际总收益: {row['total_profit']:+.2f}点")
    print(f"  交易次数: {row['trade_count']}笔")
    print(f"  胜率: {row['win_rate']:.2f}%")
    print(f"  平均每笔理论收益: {row['avg_theoretical_profit']:+.2f}点")
    print(f"  盈亏比: {row['profit_loss_ratio']:.2f}")
    print()

filtered_results = df_results[
    (df_results['win_rate'] >= 65) &
    (df_results['trade_count'] >= 20) &
    (df_results['theoretical_total_profit'] > 0)
]

if len(filtered_results) > 0:
    best = filtered_results.nlargest(1, 'theoretical_total_profit').iloc[0]
    print("\n" + "=" * 120)
    print("推荐最优组合")
    print("=" * 120)
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

df_results.to_csv('strategy_optimization_results.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 优化结果已保存到: strategy_optimization_results.csv")
if len(filtered_results) > 0:
    print(f"✅ 最优配置已保存到: best_strategy_config.json")

# 下载结果（在Colab中运行）
print("\n" + "=" * 120)
print("下载结果文件（在Colab中运行以下代码）")
print("=" * 120)
print("# from google.colab import files")
print("# files.download('strategy_optimization_results.csv')")
print("# files.download('best_strategy_config.json')")


