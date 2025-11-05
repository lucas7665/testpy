#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能稳健策略 V2 - 添加时间处理和平仓逻辑
基于CURRENT_STRATEGY_SUMMARY.md描述的策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "jqdata_fetch_and_analyze",
        os.path.join(os.path.dirname(__file__), "delete", "jqdata_fetch_and_analyze.py")
    )
    jqdata_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jqdata_module)
    
    load_data_from_database = jqdata_module.load_data_from_database
    calculate_rsi = jqdata_module.calculate_rsi
    calculate_bollinger_bands = jqdata_module.calculate_bollinger_bands
    calculate_recent_trend_vectorized = jqdata_module.calculate_recent_trend_vectorized
except Exception as e:
    print(f"⚠️  无法导入模块: {e}")
    sys.exit(1)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def can_buy_time(hour, minute):
    """
    判断当前时间是否可以买入（只做日盘）
    
    时间规则：
    - 日盘：9:30-14:30可以买入
    - 避开开盘后半小时（9:00-9:30）
    - 避开收盘前半小时（14:30-15:00）
    """
    # 日盘：9:30-14:30可以买入
    if 9 <= hour < 15:
        if hour == 9 and minute < 30:
            return False  # 开盘后半小时
        if hour == 14 and minute >= 30:
            return False  # 收盘前半小时
        return True  # 9:30-14:30可以买入
    
    return False  # 其他时间（包括夜盘）不能买入


def should_force_close(hour, minute):
    """
    判断当前时间是否应该强制平仓（只做日盘）
    
    时间规则：
    - 日盘收盘：15:00
    """
    # 日盘收盘：15:00
    if hour == 15 and minute == 0:
        return True
    
    return False


def assess_market_condition_realtime(df_day, current_idx, lookback_minutes=60):
    """
    实时评估市场状态（基于最近N分钟的数据）
    
    返回：'favorable'（有利）或 'unfavorable'（不利）
    """
    if current_idx < lookback_minutes:
        return 'favorable'  # 数据不足时默认有利
    
    # 获取最近N分钟的价格数据
    recent_prices = df_day['close'].iloc[current_idx - lookback_minutes:current_idx]
    
    if len(recent_prices) < 20:
        return 'favorable'
    
    # 1. 计算波动率
    returns = recent_prices.pct_change().dropna()
    if len(returns) == 0:
        return 'favorable'
    
    volatility = returns.std() * 100  # 转换为百分比
    
    # 2. 计算趋势强度
    price_change_pct = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
    
    # 3. 计算价格波动范围
    price_range_pct = (recent_prices.max() - recent_prices.min()) / recent_prices.mean() * 100
    
    # 4. 计算趋势一致性
    ma_short = recent_prices.tail(10).mean()
    ma_long = recent_prices.mean()
    trend_consistency = abs(ma_short - ma_long) / ma_long * 100
    
    # 判断是否不利
    high_volatility = volatility > 2.5
    strong_uptrend = (price_change_pct > 1.5 and volatility > 1.8)
    large_range = price_range_pct > 3.0
    inconsistent_trend = trend_consistency > 2.0
    
    is_unfavorable = high_volatility or strong_uptrend or large_range or inconsistent_trend
    
    return 'unfavorable' if is_unfavorable else 'favorable'


def smart_robust_strategy_single_day(df_day, date,
                                     # 基础参数
                                     take_profit=6,
                                     stop_loss=7,
                                     max_holding_bars=30,
                                     # 评分系统
                                     min_score=7.0,
                                     # 市场状态过滤
                                     use_market_filter=True,
                                     # 避开不利月份
                                     avoid_bad_periods=None,  # 格式：['2024-11', '2024-12']
                                     # 入场时机优化
                                     use_entry_timing_optimization=True):  # 是否启用入场时机优化
    """
    智能稳健策略 - 单日交易逻辑（V2：添加时间处理和平仓逻辑）
    """
    if len(df_day) < 50:
        return []
    
    # 检查是否避开不利月份
    if avoid_bad_periods:
        date_obj = pd.to_datetime(date)
        year_month = f"{date_obj.year}-{date_obj.month:02d}"
        if year_month in avoid_bad_periods:
            return []  # 避开这个月份，完全不交易
    
    df_day = df_day.copy()
    
    # 计算指标
    df_day['RSI6'] = calculate_rsi(df_day['close'], 6)
    df_day['RSI14'] = calculate_rsi(df_day['close'], 14)
    df_day['recent_up_count'] = calculate_recent_trend_vectorized(df_day['close'], 10)
    
    # 布林带
    bb_period = 20
    bb_std = 2
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        df_day['close'], period=bb_period, std=bb_std
    )
    df_day['bb_upper'] = bb_upper
    df_day['bb_middle'] = bb_middle
    
    # MACD
    macd, macd_signal, macd_hist = calculate_macd(df_day['close'])
    df_day['macd_hist'] = macd_hist
    
    # 均线
    df_day['ma5'] = df_day['close'].rolling(window=5).mean()
    df_day['ma10'] = df_day['close'].rolling(window=10).mean()
    df_day['ma20'] = df_day['close'].rolling(window=20).mean()
    
    # 转换为numpy数组
    close_arr = df_day['close'].values
    datetime_arr = df_day['datetime'].values
    rsi6_arr = df_day['RSI6'].values
    rsi14_arr = df_day['RSI14'].values
    recent_up_arr = df_day['recent_up_count'].values
    bb_upper_arr = df_day['bb_upper'].values
    bb_middle_arr = df_day['bb_middle'].values
    macd_hist_arr = df_day['macd_hist'].values
    ma5_arr = df_day['ma5'].values
    ma10_arr = df_day['ma10'].values
    ma20_arr = df_day['ma20'].values
    
    all_trades = []
    position = None
    entry_price = None
    entry_idx = None
    entry_take_profit = None  # 保存入场时的止盈止损
    entry_stop_loss = None
    
    # 遍历所有数据点（需要检查所有时间点以处理强制平仓）
    for i in range(50, len(df_day)):
        current_time = datetime_arr[i]
        if not isinstance(current_time, pd.Timestamp):
            current_time = pd.to_datetime(current_time)
        
        hour = current_time.hour
        minute = current_time.minute
        current_price = close_arr[i]
        
        # 如果有持仓，先检查是否应该强制平仓（收盘时，无论是否在允许买入时间段）
        if position is not None:
            if should_force_close(hour, minute):
                holding_bars = i - entry_idx
                profit = entry_price - current_price
                
                all_trades.append({
                    'date': date,
                    'entry_time': datetime_arr[entry_idx],
                    'exit_time': current_time,
                    'entry': entry_price,
                    'exit': current_price,
                    'profit': profit,
                    'exit_reason': '收盘平仓',
                    'holding_bars': holding_bars
                })
                position = None
                continue
            
            # 检查止盈止损（使用入场时的动态止盈止损）
            holding_bars = i - entry_idx
            profit = entry_price - current_price
            
            # 止盈
            if profit >= entry_take_profit:
                # 按止盈设置计算理论收益（不受快速波动影响）
                theoretical_profit = entry_take_profit
                all_trades.append({
                    'date': date,
                    'entry_time': datetime_arr[entry_idx],
                    'exit_time': current_time,
                    'entry': entry_price,
                    'exit': current_price,
                    'profit': profit,  # 实际收益
                    'theoretical_profit': theoretical_profit,  # 理论收益（按止盈止损设置）
                    'exit_reason': '止盈',
                    'holding_bars': holding_bars,
                    'entry_take_profit': entry_take_profit,
                    'entry_stop_loss': entry_stop_loss
                })
                position = None
                continue
            
            # 止损
            if profit <= -entry_stop_loss:
                # 按止损设置计算理论收益（不受快速波动影响）
                theoretical_profit = -entry_stop_loss
                all_trades.append({
                    'date': date,
                    'entry_time': datetime_arr[entry_idx],
                    'exit_time': current_time,
                    'entry': entry_price,
                    'exit': current_price,
                    'profit': profit,  # 实际收益
                    'theoretical_profit': theoretical_profit,  # 理论收益（按止盈止损设置）
                    'exit_reason': '止损',
                    'holding_bars': holding_bars,
                    'entry_take_profit': entry_take_profit,
                    'entry_stop_loss': entry_stop_loss
                })
                position = None
                continue
            
            # 超时
            if holding_bars > max_holding_bars:
                # 超时按实际收益计算
                theoretical_profit = profit
                all_trades.append({
                    'date': date,
                    'entry_time': datetime_arr[entry_idx],
                    'exit_time': current_time,
                    'entry': entry_price,
                    'exit': current_price,
                    'profit': profit,
                    'theoretical_profit': theoretical_profit,  # 超时按实际收益
                    'exit_reason': '超时',
                    'holding_bars': holding_bars,
                    'entry_take_profit': entry_take_profit,
                    'entry_stop_loss': entry_stop_loss
                })
                position = None
                continue
        
        # 如果没有持仓，检查是否可以买入和是否有买入信号
        # 只有在允许买入的时间段内才检查买入信号
        if position is None:
            if not can_buy_time(hour, minute):
                continue  # 不在允许买入的时间段，跳过
            
            # 如果没有持仓且可以买入，检查是否有买入信号
            # 实时评估市场状态
            if use_market_filter:
                market_condition = assess_market_condition_realtime(df_day, i, lookback_minutes=60)
                
                # 根据市场状态调整参数
                if market_condition == 'favorable':
                    required_score_local = min_score
                    take_profit_local = take_profit
                    stop_loss_local = stop_loss
                else:
                    required_score_local = min_score + 1.0  # 提高评分门槛
                    take_profit_local = take_profit - 1  # 更保守的止盈
                    stop_loss_local = stop_loss - 1  # 更保守的止损
            else:
                required_score_local = min_score
                take_profit_local = take_profit
                stop_loss_local = stop_loss
            
            # 计算信号评分
            score = 0.0
            
            # 1. RSI超买确认（3分）
            if i > 0:
                rsi6 = rsi6_arr[i]
                rsi6_prev = rsi6_arr[i-1]
                if rsi6_prev > 90 and rsi6 < rsi6_prev and rsi6 > 60:
                    score += 2.0
                    if rsi6_prev > 95:
                        score += 1.0
            
            # 2. 布林带确认（2分）
            if i >= bb_period:
                prev_price = close_arr[i-1]
                current_bb_upper = bb_upper_arr[i]
                if prev_price > current_bb_upper and current_price < current_bb_upper:
                    score += 2.0
            
            # 3. MACD确认（1.5分）
            macd_hist = macd_hist_arr[i]
            if i > 0:
                macd_hist_prev = macd_hist_arr[i-1]
                if macd_hist < 0:
                    score += 1.0
                if macd_hist_prev > 0 and macd_hist < macd_hist_prev:
                    score += 0.5
            
            # 4. 均线确认（1.5分）
            ma5 = ma5_arr[i] if pd.notna(ma5_arr[i]) else current_price
            ma10 = ma10_arr[i] if pd.notna(ma10_arr[i]) else current_price
            ma20 = ma20_arr[i] if pd.notna(ma20_arr[i]) else current_price
            
            if ma5 < ma10:
                score += 0.5
            if ma10 < ma20:
                score += 0.5
            if ma5 < ma10 < ma20:
                score += 0.5
            
            # 5. 趋势过滤（1分）
            if pd.notna(recent_up_arr[i]) and recent_up_arr[i] < 6:
                score += 1.0
            
            # 6. 价格位置（1分）
            if i >= bb_period:
                bb_middle = bb_middle_arr[i]
                if current_price > bb_middle:
                    score += 0.5
                if current_price > bb_middle * 1.002:
                    score += 0.5
            
            # 7. 市场状态检查（负分）
            if use_market_filter and i >= 20:
                prices = df_day['close'].iloc[i-20:i+1]
                ma_short = prices.tail(5).mean()
                ma_long = prices.mean()
                if ma_short > ma_long * 1.005:
                    score -= 1.0
                
                returns = prices.pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std() * 100
                    if volatility > 2.0:
                        score -= 0.5
            
            # 检查是否达到评分要求
            if score >= required_score_local:
                # ========== 入场时机优化：价格确认机制（可选）==========
                can_enter = True
                
                if use_entry_timing_optimization:
                    # 价格确认：等待价格开始下跌（或至少不再快速上涨）后再入场
                    # 基于数据分析：亏损交易入场前5分钟平均上涨0.132%，盈利交易0.078%
                    # 策略：如果价格还在快速上涨，等待（避免入场太早）
                    
                    if i >= 5:
                        prices_5min = close_arr[i-5:i+1]
                        price_momentum_5min = (prices_5min[-1] - prices_5min[0]) / prices_5min[0] * 100
                        
                        # 如果入场前5分钟价格还在快速上涨（>0.10%），等待
                        # 阈值0.10%介于盈利交易0.078%和亏损交易0.132%之间
                        if price_momentum_5min > 0.10:
                            can_enter = False
                
                # 如果价格确认通过（或未启用价格确认），入场
                if can_enter:
                    position = 'SHORT'
                    entry_price = current_price
                    entry_idx = i
                    # 使用动态调整的止盈止损
                    entry_take_profit = take_profit_local
                    entry_stop_loss = stop_loss_local
    
    return all_trades


def test_smart_robust_strategy_v2(df,
                                  take_profit=6,
                                  stop_loss=8,
                                  max_holding_bars=30,
                                  min_score=7.0,
                                  use_market_filter=True,
                                  avoid_bad_periods=['2024-11', '2024-12'],
                                  parallel=True,
                                  n_jobs=None,
                                  use_entry_timing_optimization=True):  # 是否启用入场时机优化
    """
    测试智能稳健策略 V2（并行版本）
    """
    if df is None or len(df) == 0:
        return []
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    all_dates = sorted(df['date'].unique())
    
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # 保留一个核心
    
    # 准备数据
    date_data = []
    for date in all_dates:
        df_day = df[df['date'] == date].reset_index(drop=True)
        if len(df_day) >= 50:
            date_data.append((df_day, date))
    
    if len(date_data) == 0:
        return []
    
    print(f"🚀 使用 {n_jobs} 个进程并行回测 {len(date_data)} 个交易日")
    start_time = time.time()
    
    # 创建partial函数，固定策略参数
    test_func = partial(
        smart_robust_strategy_single_day,
        take_profit=take_profit,
        stop_loss=stop_loss,
        max_holding_bars=max_holding_bars,
        min_score=min_score,
        use_market_filter=use_market_filter,
        avoid_bad_periods=avoid_bad_periods,
        use_entry_timing_optimization=use_entry_timing_optimization
    )
    
    # 并行处理
    if parallel and len(date_data) > 1:
        with Pool(processes=n_jobs) as pool:
            results = pool.starmap(test_func, date_data)
    else:
        results = [test_func(*data) for data in date_data]
    
    # 合并结果
    all_trades = []
    for trades in results:
        all_trades.extend(trades)
    
    elapsed_time = time.time() - start_time
    print(f"✅ 并行回测完成，耗时 {elapsed_time:.2f} 秒，平均每个交易日 {elapsed_time/len(date_data):.3f} 秒")
    
    return all_trades


if __name__ == '__main__':
    # 测试策略
    print("=" * 120)
    print("📊 智能稳健策略 V2 测试")
    print("=" * 120)
    
    # 加载数据
    print("\n📥 加载数据...")
    df = load_data_from_database('RB0', start_date='2024-07-27', end_date='2025-08-03')
    
    if df is None or len(df) == 0:
        print("❌ 没有可用数据")
        sys.exit(1)
    
    print(f"✅ 数据加载完成，共 {len(df)} 条记录")
    
    # 运行策略
    print("\n📈 运行策略（只做日盘）...")
    print("   参数: 评分7.0, 止盈6点, 止损8点")
    print("   时间规则: 日盘9:30-14:30（避开开盘后半小时和收盘前半小时）")
    print("   强制平仓: 日盘15:00")
    
    trades = test_smart_robust_strategy_v2(
        df,
        take_profit=6,
        stop_loss=8,
        max_holding_bars=30,
        min_score=7.0,
        use_market_filter=True,
        avoid_bad_periods=['2024-11', '2024-12'],
        parallel=True,
        n_jobs=None
    )
    
    if len(trades) == 0:
        print("\n⚠️  没有交易信号")
        sys.exit(0)
    
    # 分析结果
    df_trades = pd.DataFrame(trades)
    df_trades['date'] = pd.to_datetime(df_trades['date'])
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    
    win_trades = df_trades[df_trades['profit'] > 0]
    loss_trades = df_trades[df_trades['profit'] < 0]
    total_profit = df_trades['profit'].sum()
    win_rate = len(win_trades) / len(df_trades) * 100
    
    print("\n" + "=" * 120)
    print("📊 策略表现")
    print("=" * 120)
    print(f"\n交易次数: {len(df_trades)}笔")
    print(f"胜率: {win_rate:.2f}%")
    print(f"总收益: {total_profit:+.2f}点")
    print(f"平均每笔: {total_profit/len(df_trades):+.2f}点")
    
    # 退出原因统计
    print(f"\n退出原因:")
    exit_reasons = df_trades['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"   {reason}: {count}笔 ({count/len(df_trades)*100:.1f}%)")
    
    # 按日期统计
    daily_stats = df_trades.groupby('date').agg({
        'profit': 'sum'
    }).reset_index()
    daily_stats = daily_stats.sort_values('date')
    daily_stats['cumulative_profit'] = daily_stats['profit'].cumsum()
    
    print(f"\n盈利天数: {len(daily_stats[daily_stats['profit'] > 0])}天")
    print(f"亏损天数: {len(daily_stats[daily_stats['profit'] < 0])}天")
    
    # 检查收盘平仓
    close_orders = df_trades[df_trades['exit_reason'] == '收盘平仓']
    print(f"\n收盘平仓: {len(close_orders)}笔")
    if len(close_orders) > 0:
        print(f"   日盘收盘平仓（15:00）: {len(close_orders[close_orders['exit_time'].dt.hour == 15])}笔")
    
    # 验证时间规则
    print(f"\n时间规则验证:")
    entry_times = df_trades['entry_time'].dt.hour
    entry_minutes = df_trades['entry_time'].dt.minute
    # 检查是否有在禁止时段买入的
    invalid_buy = df_trades[(entry_times == 9) & (entry_minutes < 30)]
    invalid_buy2 = df_trades[(entry_times == 14) & (entry_minutes >= 30)]
    invalid_buy3 = df_trades[(entry_times < 9) | (entry_times >= 15)]
    invalid_buy4 = df_trades[(entry_times >= 21)]
    if len(invalid_buy) == 0 and len(invalid_buy2) == 0 and len(invalid_buy3) == 0 and len(invalid_buy4) == 0:
        print(f"   ✅ 所有买入都在允许时段（9:30-14:30）")
    else:
        print(f"   ⚠️  发现异常买入时间: 开盘后半小时{len(invalid_buy)}笔, 收盘前半小时{len(invalid_buy2)}笔, 其他时段{len(invalid_buy3)+len(invalid_buy4)}笔")
    
    # 按月统计
    df_trades['year_month'] = df_trades['date'].dt.to_period('M')
    monthly_stats = df_trades.groupby('year_month').agg({
        'profit': ['sum', 'count']
    }).reset_index()
    monthly_stats.columns = ['year_month', 'monthly_profit', 'trade_count']
    monthly_stats = monthly_stats.sort_values('year_month')
    monthly_stats['cumulative_profit'] = monthly_stats['monthly_profit'].cumsum()
    
    print(f"\n📅 按月收益:")
    print("-" * 80)
    print(f"{'月份':<12} {'交易次数':<10} {'月收益':<12} {'累计收益':<12} {'评价':<10}")
    print("-" * 80)
    for _, row in monthly_stats.iterrows():
        month_str = str(row['year_month'])
        profit = row['monthly_profit']
        count = int(row['trade_count'])
        cum = row['cumulative_profit']
        status = "✅ 盈利" if profit > 0 else "❌ 亏损" if profit < 0 else "⚪ 持平"
        print(f"{month_str:<12} {count:<10} {profit:<11.2f} {cum:<11.2f} {status:<10}")
    
    loss_months = len(monthly_stats[monthly_stats['monthly_profit'] < 0])
    print(f"\n亏损月份: {loss_months}个月")
    print(f"盈利月份: {len(monthly_stats) - loss_months}个月")

