#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
螺纹钢实时RSI监控
专注于当天的实时分钟级数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import time

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告：akshare未安装，请运行: pip install akshare")


def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_multi_rsi(prices):
    """
    计算同花顺式的多周期RSI
    返回RSI6, RSI12, RSI24
    """
    rsi6 = calculate_rsi(prices, period=6)
    rsi12 = calculate_rsi(prices, period=12)
    rsi24 = calculate_rsi(prices, period=24)
    return rsi6, rsi12, rsi24


def get_realtime_rebar_data():
    """
    获取螺纹钢当天实时分钟数据
    优先使用1分钟数据，如果失败则使用5分钟数据
    """
    if not AKSHARE_AVAILABLE:
        print("❌ 错误：需要安装akshare库")
        print("请运行: pip install akshare")
        return None
    
    # 尝试不同的数据周期
    periods = ['1min', '5min', '15min']
    
    for period in periods:
        try:
            print(f"正在获取{period}数据...", end=" ")
            
            # 获取螺纹钢主力合约的分钟数据
            df = ak.futures_zh_minute_sina(symbol="RB0", period=period)
            
            if df is not None and not df.empty:
                print(f"✓ 成功获取{len(df)}条数据")
                
                # 确保数据是今天的
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    today = date.today()
                    # 过滤今天的数据
                    df_today = df[df['datetime'].dt.date == today]
                    
                    if not df_today.empty:
                        print(f"✓ 今天的数据有{len(df_today)}条")
                        return df_today, period
                    else:
                        print(f"⚠️  没有今天的数据，返回最新数据")
                        return df.tail(100), period  # 返回最近100条
                
                return df, period
                
        except Exception as e:
            print(f"✗ 失败: {e}")
            continue
    
    # 如果所有分钟数据都失败，尝试获取实时报价
    try:
        print("尝试获取实时报价数据...", end=" ")
        df = ak.futures_zh_spot(symbol="螺纹钢")
        if df is not None and not df.empty:
            print("✓ 成功")
            return df, 'spot'
    except Exception as e:
        print(f"✗ 失败: {e}")
    
    print("❌ 所有数据源都失败")
    return None, None


def show_realtime_rsi(threshold=30):
    """显示当天实时RSI（同花顺式多周期）"""
    
    print("=" * 80)
    print("🔍 螺纹钢实时RSI监控（同花顺式）")
    print("=" * 80)
    
    result = get_realtime_rebar_data()
    
    if result[0] is None:
        print("❌ 无法获取数据")
        return
    
    df, period = result
    
    # 计算多周期RSI（同花顺式）
    df['RSI6'], df['RSI12'], df['RSI24'] = calculate_multi_rsi(df['close'])
    # 保留RSI14作为参考
    df['RSI'] = df['RSI6']  # 使用RSI6作为主要指标
    
    # 显示数据信息
    print(f"\n📊 数据信息:")
    print(f"   数据周期: {period}")
    print(f"   数据条数: {len(df)}")
    
    if 'datetime' in df.columns:
        print(f"   时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    
    # 显示最近的数据（同花顺式多周期RSI）
    print(f"\n📈 最近数据（最新10条）:")
    print("-" * 100)
    
    # 选择要显示的列
    display_df = df.tail(10).copy()
    
    if 'datetime' in display_df.columns:
        display_df['时间'] = display_df['datetime'].dt.strftime('%H:%M')
    
    # 重命名列
    display_df = display_df.rename(columns={
        'open': '开盘',
        'high': '最高',
        'low': '最低',
        'close': '收盘',
        'volume': '成交量',
    })
    
    # 格式化显示
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    # 显示多周期RSI
    display_cols = []
    if '时间' in display_df.columns:
        display_cols.append('时间')
    display_cols.extend(['开盘', '最高', '最低', '收盘', 'RSI6', 'RSI12', 'RSI24'])
    
    available_cols = [c for c in display_cols if c in display_df.columns]
    print(display_df[available_cols].to_string(index=False))
    print("-" * 100)
    
    # 获取最新数据
    latest = df.iloc[-1]
    latest_rsi6 = latest['RSI6']
    latest_rsi12 = latest['RSI12']
    latest_rsi24 = latest['RSI24']
    latest_price = latest['close']
    
    # 计算涨跌
    if len(df) > 1:
        prev_price = df['close'].iloc[-2]
        change = latest_price - prev_price
        change_pct = (change / prev_price) * 100
    else:
        change = 0
        change_pct = 0
    
    # 显示当前状态（同花顺式）
    print(f"\n💹 当前行情 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):")
    print(f"   最新价格: {latest_price:.2f}")
    print(f"   涨跌幅度: {change:+.2f} ({change_pct:+.2f}%)")
    print(f"   最高价格: {latest['high']:.2f}")
    print(f"   最低价格: {latest['low']:.2f}")
    print(f"\n📊 RSI指标（同花顺式）:")
    print(f"   RSI(6):  {latest_rsi6:.2f}  ← 短线指标（最灵敏）")
    print(f"   RSI(12): {latest_rsi12:.2f}  ← 中线指标")
    print(f"   RSI(24): {latest_rsi24:.2f}  ← 长线指标（最稳定）")
    
    # RSI综合分析（基于RSI6）
    print(f"\n💡 交易建议（基于RSI6）:")
    if pd.isna(latest_rsi6):
        print("   ⚠️  RSI数据不足，需要更多历史数据")
    elif latest_rsi6 < threshold:
        print(f"   ✅ RSI6 = {latest_rsi6:.2f} < {threshold}")
        print(f"   🔔 【强烈买入信号】RSI6进入超卖区域!")
        
        # 多周期确认
        if latest_rsi12 < 40:
            print(f"   ✅ RSI12 = {latest_rsi12:.2f} < 40，中线也确认超卖")
            print(f"   💰 【高置信度买入】多周期共振，买入信号更强!")
        else:
            print(f"   ⚠️  RSI12 = {latest_rsi12:.2f}，中线未确认")
            print(f"   💡 建议：谨慎买入，可能只是短线反弹")
            
    elif latest_rsi6 > 70:
        print(f"   ⚠️  RSI6 = {latest_rsi6:.2f} > 70")
        print(f"   【卖出信号】RSI6进入超买区域!")
        
        # 多周期确认
        if latest_rsi12 > 60:
            print(f"   ⚠️  RSI12 = {latest_rsi12:.2f} > 60，中线也确认超买")
            print(f"   💰 【高置信度卖出】多周期共振，卖出信号更强!")
        else:
            print(f"   ✅ RSI12 = {latest_rsi12:.2f}，中线未确认")
            print(f"   💡 建议：考虑部分止盈，留仓观察")
            
    elif latest_rsi6 < 40:
        print(f"   📉 RSI6 = {latest_rsi6:.2f} (接近超卖区)")
        print(f"   💡 建议：关注价格走势，准备买入")
        
    elif latest_rsi6 > 60:
        print(f"   📈 RSI6 = {latest_rsi6:.2f} (接近超买区)")
        print(f"   💡 建议：关注价格走势，考虑止盈")
        
    else:
        print(f"   ⏳ RSI6 = {latest_rsi6:.2f} (正常区间 30-70)")
        print(f"   💡 建议：继续观望，等待明确信号")
    
    print("\n" + "=" * 100)


def monitor_loop(threshold=30, interval=60):
    """持续监控模式（同花顺式多周期RSI）"""
    
    print("=" * 100)
    print("🔄 开始实时监控螺纹钢RSI（同花顺式）")
    print(f"   买入阈值: RSI6 < {threshold}")
    print(f"   刷新间隔: {interval}秒")
    print(f"   按 Ctrl+C 停止监控")
    print("=" * 100)
    print()
    
    while True:
        try:
            result = get_realtime_rebar_data()
            
            if result[0] is not None:
                df, period = result
                
                # 计算多周期RSI
                df['RSI6'], df['RSI12'], df['RSI24'] = calculate_multi_rsi(df['close'])
                
                # 获取最新数据
                latest_rsi6 = df['RSI6'].iloc[-1]
                latest_rsi12 = df['RSI12'].iloc[-1]
                latest_rsi24 = df['RSI24'].iloc[-1]
                latest_price = df['close'].iloc[-1]
                
                # 计算涨跌
                if len(df) > 1:
                    prev_price = df['close'].iloc[-2]
                    change = latest_price - prev_price
                    change_pct = (change / prev_price) * 100
                else:
                    change = 0
                    change_pct = 0
                
                # 显示当前状态
                current_time = datetime.now().strftime("%H:%M:%S")
                status = f"[{current_time}] 价格:{latest_price:.2f} ({change:+.2f},{change_pct:+.2f}%) | RSI6:{latest_rsi6:.2f} RSI12:{latest_rsi12:.2f} RSI24:{latest_rsi24:.2f}"
                
                # 判断信号
                if pd.notna(latest_rsi6):
                    if latest_rsi6 < threshold:
                        if latest_rsi12 < 40:
                            print(f"{status} 🔔🔔 【强烈买入】多周期共振!")
                        else:
                            print(f"{status} 🔔 【买入信号】")
                    elif latest_rsi6 > 70:
                        if latest_rsi12 > 60:
                            print(f"{status} ⚠️⚠️  【强烈卖出】多周期共振!")
                        else:
                            print(f"{status} ⚠️  【卖出信号】")
                    else:
                        print(f"{status} ✓")
                else:
                    print(f"{status} (RSI数据不足)")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 获取数据失败")
            
            # 等待
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            # 持续监控模式
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            monitor_loop(threshold=threshold, interval=interval)
        elif sys.argv[1] == "once":
            # 单次查看
            threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            show_realtime_rsi(threshold=threshold)
        else:
            print("用法:")
            print("  python realtime_rsi_monitor.py once [阈值]     # 单次查看")
            print("  python realtime_rsi_monitor.py monitor [间隔] [阈值]  # 持续监控")
    else:
        # 默认单次查看
        show_realtime_rsi(threshold=30)

