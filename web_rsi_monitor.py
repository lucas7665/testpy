#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
螺纹钢RSI Web监控页面
实时显示RSI指标和K线图
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import json

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

app = Flask(__name__)


def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_rebar_data():
    """获取螺纹钢实时数据"""
    if not AKSHARE_AVAILABLE:
        # 返回模拟数据
        return generate_mock_data()
    
    try:
        # 尝试获取1分钟数据
        df = ak.futures_zh_minute_sina(symbol="RB0", period="1min")
        
        if df is not None and not df.empty:
            # 过滤今天的数据
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                today = pd.Timestamp.now().date()
                df_today = df[df['datetime'].dt.date == today]
                
                if not df_today.empty:
                    return df_today
            
            return df.tail(240)  # 返回最近240条（约4小时）
            
    except Exception as e:
        print(f"获取数据失败: {e}")
    
    # 如果失败，返回模拟数据
    return generate_mock_data()


def generate_mock_data():
    """生成模拟数据（用于演示）"""
    import random
    from datetime import timedelta
    
    now = datetime.now()
    data = []
    base_price = 3130
    
    for i in range(240):
        time = now - timedelta(minutes=240-i)
        price = base_price + random.uniform(-10, 10) + np.sin(i/20) * 5
        
        data.append({
            'datetime': time,
            'open': price + random.uniform(-2, 2),
            'high': price + random.uniform(0, 3),
            'low': price - random.uniform(0, 3),
            'close': price,
            'volume': random.randint(1000, 5000)
        })
    
    df = pd.DataFrame(data)
    return df


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    """API: 获取实时数据"""
    try:
        print("收到数据请求...")  # 调试信息
        
        # 获取数据
        df = get_rebar_data()
        
        print(f"获取到 {len(df) if df is not None else 0} 条数据")  # 调试信息
        
        if df is None or df.empty:
            print("数据为空，返回错误")
            return jsonify({'success': False, 'error': '无法获取数据'}), 500
        
        # 计算多周期RSI
        df['RSI6'] = calculate_rsi(df['close'], period=6)
        df['RSI12'] = calculate_rsi(df['close'], period=12)
        df['RSI24'] = calculate_rsi(df['close'], period=24)
        
        # 准备返回数据
        latest = df.iloc[-1]
        
        # K线数据（最近100条）
        chart_data = df.tail(100).copy()
        
        if 'datetime' in chart_data.columns:
            chart_data['time'] = chart_data['datetime'].dt.strftime('%H:%M')
        else:
            chart_data['time'] = [f"{i:02d}:{j:02d}" for i, j in enumerate(range(len(chart_data)))]
        
        # 计算涨跌
        if len(df) > 1:
            prev_price = df['close'].iloc[-2]
            change = latest['close'] - prev_price
            change_pct = (change / prev_price) * 100
        else:
            change = 0
            change_pct = 0
        
        # 构建响应数据（确保所有数据都可JSON序列化）
        def safe_float(value):
            """安全转换为float"""
            if pd.isna(value) or value is None:
                return None
            try:
                return float(value)
            except:
                return None
        
        def safe_int(value):
            """安全转换为int"""
            if pd.isna(value) or value is None:
                return 0
            try:
                return int(value)
            except:
                return 0
        
        response = {
            'success': True,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current': {
                'price': safe_float(latest['close']),
                'high': safe_float(latest['high']),
                'low': safe_float(latest['low']),
                'change': safe_float(change),
                'change_pct': safe_float(change_pct),
                'volume': safe_int(latest['volume']),
                'rsi6': safe_float(latest['RSI6']),
                'rsi12': safe_float(latest['RSI12']),
                'rsi24': safe_float(latest['RSI24']),
            },
            'chart': {
                'time': [str(t) for t in chart_data['time'].tolist()],
                'open': [safe_float(x) for x in chart_data['open'].tolist()],
                'high': [safe_float(x) for x in chart_data['high'].tolist()],
                'low': [safe_float(x) for x in chart_data['low'].tolist()],
                'close': [safe_float(x) for x in chart_data['close'].tolist()],
                'volume': [safe_int(x) for x in chart_data['volume'].tolist()],
                'rsi6': [safe_float(x) for x in chart_data['RSI6'].tolist()],
                'rsi12': [safe_float(x) for x in chart_data['RSI12'].tolist()],
                'rsi24': [safe_float(x) for x in chart_data['RSI24'].tolist()],
            }
        }
        
        print("成功返回数据")  # 调试信息
        return jsonify(response)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")  # 调试信息
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # 使用5001端口（避免与Mac的AirPlay和其他服务冲突）
    port = 5001
    
    print("=" * 80)
    print("🌐 螺纹钢RSI Web监控系统")
    print("=" * 80)
    print("\n启动Web服务器...")
    print(f"\n请在浏览器中访问: http://localhost:{port}")
    print("\n按 Ctrl+C 停止服务器\n")
    print("=" * 80)
    
    # 启动Flask服务器
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ 错误: 端口 {port} 也被占用了")
            print("\n正在尝试其他端口...")
            # 尝试其他端口
            for try_port in [5002, 5003, 8888, 9999, 3000]:
                try:
                    print(f"尝试端口 {try_port}...")
                    app.run(host='0.0.0.0', port=try_port, debug=False)
                    break
                except OSError:
                    continue
        else:
            raise

