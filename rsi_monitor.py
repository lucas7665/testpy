#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
螺纹钢RSI监控脚本
当RSI < 30时发出买入提醒
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time

# 需要安装的库：
# pip install pandas numpy akshare

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告：akshare未安装，请运行: pip install akshare")


def calculate_rsi(data, period=14):
    """
    计算RSI指标
    
    参数:
        data: 价格数据（Series）
        period: RSI周期，默认14
    
    返回:
        RSI值
    """
    delta = data.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def get_rebar_realtime_data(period='1min'):
    """
    获取螺纹钢期货实时分钟级数据
    
    参数:
        period: 数据周期 '1min', '5min', '15min', '30min', '60min'
    
    返回:
        DataFrame: 包含实时价格数据
    """
    if not AKSHARE_AVAILABLE:
        print("错误：需要安装akshare库")
        return None
    
    try:
        # 获取螺纹钢实时行情数据
        # 方法1: 获取实时tick数据
        print("正在获取螺纹钢实时数据...")
        
        # 尝试获取分钟线数据
        try:
            # 获取主力合约代码
            # RB0 代表螺纹钢主力合约
            df = ak.futures_zh_minute_sina(symbol="RB0", period=period)
            
            if df is not None and not df.empty:
                # 重命名列以便统一处理
                df.rename(columns={
                    'datetime': 'time',
                    'close': 'close',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'volume': 'volume'
                }, inplace=True)
                
                return df
        except Exception as e:
            print(f"获取分钟数据失败，尝试其他方式: {e}")
        
        # 方法2: 如果分钟数据获取失败，使用日线数据
        try:
            df = ak.futures_zh_daily_sina(symbol="RB0")
            if df is not None and not df.empty:
                print("注意：使用日线数据代替分钟数据")
                return df
        except Exception as e:
            print(f"获取日线数据也失败: {e}")
        
        print("所有数据获取方式都失败")
        return None
        
    except Exception as e:
        print(f"获取数据时出错: {e}")
        return None


def get_rebar_data():
    """
    获取螺纹钢期货数据（保持兼容性）
    默认获取实时分钟数据
    
    返回:
        DataFrame: 包含价格数据
    """
    return get_rebar_realtime_data(period='1min')


def monitor_rsi(threshold=30, interval=60):
    """
    监控RSI值，当小于阈值时提醒
    
    参数:
        threshold: RSI阈值，默认30
        interval: 检查间隔（秒），默认60秒
    """
    print(f"开始监控螺纹钢RSI...")
    print(f"买入阈值: RSI < {threshold}")
    print(f"检查间隔: {interval}秒")
    print("-" * 60)
    
    while True:
        try:
            # 获取数据
            df = get_rebar_data()
            
            if df is not None and not df.empty:
                # 计算RSI
                df['RSI'] = calculate_rsi(df['close'], period=14)
                
                # 获取最新RSI值
                latest_rsi = df['RSI'].iloc[-1]
                latest_price = df['close'].iloc[-1]
                
                # 获取时间信息
                if 'time' in df.columns:
                    latest_time = df['time'].iloc[-1]
                elif 'date' in df.columns:
                    latest_time = df['date'].iloc[-1]
                else:
                    latest_time = datetime.now()
                
                # 计算涨跌
                if len(df) > 1:
                    prev_price = df['close'].iloc[-2]
                    change = latest_price - prev_price
                    change_pct = (change / prev_price) * 100
                    change_str = f"{change:+.2f} ({change_pct:+.2f}%)"
                else:
                    change_str = "N/A"
                
                # 显示当前状态
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] 价格: {latest_price:.2f} [{change_str}], RSI: {latest_rsi:.2f}", end="")
                
                # 判断是否触发买入信号
                if latest_rsi < threshold:
                    print(f" 🔔 【买入信号】RSI已低于{threshold}!")
                    # 这里可以添加通知功能，如发送邮件、微信等
                    send_notification(latest_price, latest_rsi, change_str)
                else:
                    print(f" ✓ 正常")
                
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 未能获取数据")
            
            # 等待指定时间后再次检查
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(interval)


def send_notification(price, rsi, change_str="N/A"):
    """
    发送买入通知
    可以根据需要实现邮件、微信、钉钉等通知方式
    """
    message = f"""
    ==========================================
    🔔 螺纹钢买入信号！
    ==========================================
    时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    价格: {price:.2f}
    涨跌: {change_str}
    RSI:  {rsi:.2f}
    建议: RSI进入超卖区域，考虑买入
    ==========================================
    """
    print(message)
    
    # TODO: 在这里添加您需要的通知方式
    # 例如：发送邮件、推送到手机等


def check_once(threshold=30, show_detail=True):
    """
    单次检查RSI值
    
    参数:
        threshold: RSI阈值
        show_detail: 是否显示详细数据
    """
    print("正在获取螺纹钢实时数据...")
    
    df = get_rebar_data()
    
    if df is not None and not df.empty:
        # 计算RSI
        df['RSI'] = calculate_rsi(df['close'], period=14)
        
        if show_detail:
            # 显示当天最近的数据
            print("\n📊 螺纹钢最近数据（实时更新）:")
            print("-" * 80)
            
            # 根据数据类型选择显示列
            if 'time' in df.columns:
                # 分钟数据
                display_cols = ['time', 'open', 'high', 'low', 'close', 'RSI']
                recent_data = df[display_cols].tail(10)
                print("最近10分钟数据:")
            elif 'date' in df.columns:
                # 日线数据
                display_cols = ['date', 'open', 'high', 'low', 'close', 'RSI']
                recent_data = df[display_cols].tail(5)
                print("最近5天数据:")
            else:
                display_cols = ['open', 'high', 'low', 'close', 'RSI']
                recent_data = df[display_cols].tail(10)
            
            # 格式化显示
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            print(recent_data.to_string(index=False))
            print("-" * 80)
        
        # 获取最新数据
        latest_rsi = df['RSI'].iloc[-1]
        latest_price = df['close'].iloc[-1]
        latest_high = df['high'].iloc[-1] if 'high' in df.columns else latest_price
        latest_low = df['low'].iloc[-1] if 'low' in df.columns else latest_price
        
        # 计算涨跌
        if len(df) > 1:
            prev_price = df['close'].iloc[-2]
            change = latest_price - prev_price
            change_pct = (change / prev_price) * 100
        else:
            change = 0
            change_pct = 0
        
        # 显示当前状态
        print(f"\n📈 当前螺纹钢行情:")
        print(f"   最新价格: {latest_price:.2f}")
        print(f"   涨跌幅度: {change:+.2f} ({change_pct:+.2f}%)")
        print(f"   最高价格: {latest_high:.2f}")
        print(f"   最低价格: {latest_low:.2f}")
        print(f"   当前RSI:  {latest_rsi:.2f}")
        
        # 判断买入信号
        print(f"\n💡 操作建议:")
        if latest_rsi < threshold:
            print(f"   ✅ RSI = {latest_rsi:.2f} < {threshold}")
            print(f"   🔔 【建议买入】RSI进入超卖区域，可能反弹!")
        elif latest_rsi > 70:
            print(f"   ⚠️  RSI = {latest_rsi:.2f} > 70")
            print(f"   【建议卖出】RSI进入超买区域，可能回调!")
        else:
            print(f"   ⏳ RSI = {latest_rsi:.2f} (30-70区间)")
            print(f"   【观望】等待RSI进入超卖或超买区域")
        
        print()
    else:
        print("❌ 获取数据失败，请检查网络连接或稍后重试")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("螺纹钢RSI监控工具")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        # 单次检查模式
        check_once(threshold=30)
    else:
        # 持续监控模式
        print("\n模式选择:")
        print("1. 持续监控（每分钟检查一次）")
        print("2. 单次检查")
        
        choice = input("\n请选择模式 (1/2): ").strip()
        
        if choice == "2":
            check_once(threshold=30)
        else:
            monitor_rsi(threshold=30, interval=60)

