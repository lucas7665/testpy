#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用JQData API获取螺纹钢期货分钟数据并保存到数据库
然后分析策略盈利情况
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pymysql
import time
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    import jqdatasdk as jq
    JQDATA_AVAILABLE = True
except ImportError:
    JQDATA_AVAILABLE = False
    print("⚠️  未安装jqdatasdk，请运行: pip install jqdatasdk")

# 数据库配置
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'pig',
    'charset': 'utf8mb4'
}

# JQData配置
JQ_USERNAME = '15864005520'
JQ_PASSWORD = '2011201644Aa.'


def get_db_connection():
    """获取数据库连接"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return None


def create_table_if_not_exists():
    """创建数据表（如果不存在）"""
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        # 创建表
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS futures_minute_data_jq (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL COMMENT '合约代码',
            datetime DATETIME NOT NULL COMMENT '时间',
            date DATE NOT NULL COMMENT '日期',
            open DECIMAL(10, 2) NOT NULL COMMENT '开盘价',
            high DECIMAL(10, 2) NOT NULL COMMENT '最高价',
            low DECIMAL(10, 2) NOT NULL COMMENT '最低价',
            close DECIMAL(10, 2) NOT NULL COMMENT '收盘价',
            volume BIGINT NOT NULL COMMENT '成交量',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
            UNIQUE KEY uk_symbol_datetime (symbol, datetime),
            KEY idx_date (date),
            KEY idx_symbol_date (symbol, date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='期货分钟数据表（JQData）';
        """
        cursor.execute(create_table_sql)
        conn.commit()
        print("✅ 数据表创建/检查完成")
        return True
    except Exception as e:
        print(f"❌ 创建表失败: {e}")
        return False
    finally:
        conn.close()


def login_jqdata(username, password):
    """登录JQData"""
    if not JQDATA_AVAILABLE:
        print("❌ jqdatasdk未安装")
        return False
    
    try:
        jq.auth(username, password)
        print(f"✅ JQData登录成功")
        return True
    except Exception as e:
        print(f"❌ JQData登录失败: {e}")
        return False


def get_futures_data_from_jqdata(symbol='RB0', start_date=None, end_date=None):
    """
    从JQData获取期货分钟数据
    
    参数:
        symbol: 合约代码，如'RB0'（螺纹钢主力合约）
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'
    
    返回:
        DataFrame
    """
    if not JQDATA_AVAILABLE:
        return None
    
    try:
        # JQData中螺纹钢期货合约代码格式
        # 主力合约通常是 rb9999.XSGE 或 rb8888.XSGE
        # 也可以使用 get_dominant_future 获取主力合约
        
        if symbol == 'RB0':
            try:
                # 方法1: 尝试使用get_dominant_future获取主力合约
                # 注意：需要根据实际日期获取对应时期的主力合约
                # 这里我们使用一个中间日期来获取主力合约
                import datetime as dt
                mid_date = dt.datetime.strptime(end_date, '%Y-%m-%d') if end_date else dt.datetime.now()
                
                # 获取主力合约代码
                try:
                    dominant = jq.get_dominant_future('RB', mid_date)
                    if dominant:
                        jq_symbol = dominant
                        print(f"📊 使用主力合约: {jq_symbol}")
                    else:
                        raise Exception("未获取到主力合约")
                except:
                    # 方法2: 直接使用rb9999.XSGE（主力合约通用代码）
                    jq_symbol = 'rb9999.XSGE'
                    print(f"📊 使用合约代码: {jq_symbol}")
            except Exception as e:
                print(f"⚠️  获取合约代码失败，使用rb9999.XSGE: {e}")
                jq_symbol = 'rb9999.XSGE'
        else:
            jq_symbol = symbol
        
        # 获取分钟数据
        print(f"📥 正在获取数据: {jq_symbol} from {start_date} to {end_date}")
        
        # 使用get_price获取分钟数据
        # 注意：期货数据不需要fq参数
        # JQData的get_price返回的DataFrame通常以DatetimeIndex为索引
        df = jq.get_price(
            jq_symbol,
            start_date=start_date,
            end_date=end_date,
            frequency='1m',
            fields=['open', 'high', 'low', 'close', 'volume'],
            skip_paused=True
        )
        
        if df is None or len(df) == 0:
            print("❌ 未获取到数据")
            return None
        
        print(f"📊 原始数据形状: {df.shape}")
        print(f"📊 原始数据列: {df.columns.tolist()}")
        print(f"📊 原始数据索引类型: {type(df.index)}")
        print(f"📊 原始数据索引名称: {df.index.name}")
        
        # JQData返回的DataFrame以DatetimeIndex为索引
        # 将索引转换为datetime列
        if isinstance(df.index, pd.DatetimeIndex):
            # 索引是DatetimeIndex，将其转换为列
            df = df.reset_index()
            # reset_index后，时间索引会变成第一列，但列名可能是None或'time'
            # 检查第一列是否是时间类型
            first_col = df.columns[0]
            if first_col is None or first_col == 'index' or pd.api.types.is_datetime64_any_dtype(df[first_col]):
                # 重命名第一列为datetime
                df = df.rename(columns={first_col: 'datetime'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'datetime'})
        else:
            # 索引不是DatetimeIndex，但可能有时间列
            df = df.reset_index()
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'datetime'})
            elif 'datetime' not in df.columns:
                print(f"❌ 无法找到时间列，当前列: {df.columns.tolist()}")
                return None
        
        # 确保有datetime列
        if 'datetime' not in df.columns:
            print(f"❌ 无法找到时间列，当前列: {df.columns.tolist()}")
            return None
        
        # 检查必需的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️  缺少列: {missing_columns}")
            print(f"   当前列: {df.columns.tolist()}")
        
        # 添加日期列
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date
        
        # 添加symbol列
        df['symbol'] = symbol
        
        # 确保数据按时间排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"✅ 获取到 {len(df)} 条数据")
        if len(df) > 0:
            print(f"   日期范围: {df['date'].min()} 至 {df['date'].max()}")
            print(f"   时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_data_to_database(df, symbol='RB0', batch_size=1000):
    """保存数据到数据库（批量优化版本）"""
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        
        inserted = 0
        updated = 0
        
        # 批量插入
        values_list = []
        for idx, row in df.iterrows():
            values_list.append((
                symbol,
                row['datetime'],
                row['date'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            ))
            
            # 每batch_size条执行一次
            if len(values_list) >= batch_size:
                sql = """
                INSERT INTO futures_minute_data_jq 
                (symbol, datetime, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    open = VALUES(open),
                    high = VALUES(high),
                    low = VALUES(low),
                    close = VALUES(close),
                    volume = VALUES(volume),
                    updated_at = CURRENT_TIMESTAMP
                """
                try:
                    results = cursor.executemany(sql, values_list)
                    # executemany 返回受影响的行数总和
                    # 注意：MySQL的executemany对于ON DUPLICATE KEY UPDATE
                    # 可能无法准确区分插入和更新，所以这里简化统计
                    inserted += len(values_list)
                    values_list = []
                except Exception as e:
                    print(f"⚠️  批量保存失败: {e}")
                    values_list = []
        
        # 处理剩余数据
        if values_list:
            sql = """
            INSERT INTO futures_minute_data_jq 
            (symbol, datetime, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume),
                updated_at = CURRENT_TIMESTAMP
            """
            try:
                cursor.executemany(sql, values_list)
                inserted += len(values_list)
            except Exception as e:
                print(f"⚠️  批量保存失败: {e}")
        
        conn.commit()
        print(f"✅ 数据保存完成: 处理 {inserted} 条记录")
        return True
        
    except Exception as e:
        print(f"❌ 保存数据失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def load_data_from_database(symbol='RB0', start_date=None, end_date=None):
    """从数据库加载数据"""
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        sql = "SELECT * FROM futures_minute_data_jq WHERE symbol = %s"
        params = [symbol]
        
        if start_date:
            sql += " AND date >= %s"
            params.append(start_date)
        
        if end_date:
            sql += " AND date <= %s"
            params.append(end_date)
        
        sql += " ORDER BY datetime ASC"
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        if len(results) == 0:
            print("⚠️  数据库中没有数据")
            return None
        
        df = pd.DataFrame(results)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # 确保价格字段是float类型（数据库返回的可能是Decimal）
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 确保volume是整数类型
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('Int64')
        
        print(f"✅ 从数据库加载 {len(df)} 条数据")
        print(f"   日期范围: {df['date'].min()} 至 {df['date'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None
    finally:
        conn.close()


def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices, period=20, std=2):
    """计算布林带"""
    sma = prices.rolling(window=period).mean()
    std_val = prices.rolling(window=period).std()
    upper = sma + std_val * std
    lower = sma - std_val * std
    return upper, sma, lower


def calculate_recent_trend_vectorized(prices, period=10):
    """计算最近N根K线的上涨数（向量化版本，速度更快）"""
    if len(prices) < period:
        return pd.Series(np.zeros(len(prices)), index=prices.index if hasattr(prices, 'index') else None)
    
    # 转换为numpy数组以提高性能
    prices_arr = np.asarray(prices)
    diff = np.diff(prices_arr)  # diff[i] = prices[i+1] - prices[i]
    
    # 对于位置i，需要看的是i-period到i-1的diff，即diff[i-period:i]
    # 使用滑动窗口求和
    up_count = np.zeros(len(prices))
    
    # 对于位置i >= period，计算diff[i-period:i]中大于0的数量
    # 使用numpy的convolve或者直接循环，但用numpy加速
    for i in range(period, len(prices)):
        # diff的索引i-period对应prices的i-period到i-period+1的变化
        # 我们需要看的是i-period到i-1的变化，即diff[i-period:i-1]
        window_diff = diff[max(0, i-period):i]
        up_count[i] = np.sum(window_diff > 0)
    
    return pd.Series(up_count, index=prices.index if hasattr(prices, 'index') else None)


def bollinger_signal(df, idx, bb_period=20, bb_std=2):
    """布林带信号"""
    if idx < bb_period + 1:
        return 'HOLD'
    sma = df['close'].iloc[idx-bb_period:idx].mean()
    std = df['close'].iloc[idx-bb_period:idx].std()
    upper_band = sma + bb_std * std
    current_price = df['close'].iloc[idx]
    prev_price = df['close'].iloc[idx-1]
    if prev_price > upper_band and current_price < upper_band:
        return 'SELL'
    return 'HOLD'


def rsi_signal_sell(df, idx, sell_high=90, sell_low=60):
    """RSI卖空信号"""
    if idx < 30:
        return 'HOLD'
    rsi6 = df['RSI6'].iloc[idx]
    rsi6_prev = df['RSI6'].iloc[idx-1]
    if rsi6_prev > sell_high and rsi6 < rsi6_prev and rsi6 > sell_low:
        return 'SELL'
    return 'HOLD'


def test_strategy_single_day(df_day, date, rsi_sell_high=90, rsi_sell_low=60,
                             take_profit=8, stop_loss=5, filter_threshold=7):
    """测试单日策略（优化版本，用于多进程）"""
    if len(df_day) < 50:
        return []
    
    # 预先计算所有指标（向量化）
    df_day = df_day.copy()
    df_day['RSI6'] = calculate_rsi(df_day['close'], 6)
    df_day['recent_up_count'] = calculate_recent_trend_vectorized(df_day['close'], 10)
    
    # 预先计算布林带（向量化）
    bb_period = 20
    bb_std = 2
    bb_upper = df_day['close'].rolling(window=bb_period).mean() + df_day['close'].rolling(window=bb_period).std() * bb_std
    
    # 转换为numpy数组以提高访问速度
    close_arr = df_day['close'].values
    datetime_arr = df_day['datetime'].values
    rsi6_arr = df_day['RSI6'].values
    recent_up_arr = df_day['recent_up_count'].values
    bb_upper_arr = bb_upper.values
    
    all_trades = []
    position = None
    entry_price = None
    entry_idx = None
    
    # 只处理白天时段（9:00-15:00）的索引
    trading_indices = []
    for i in range(50, len(df_day)):
        current_time = datetime_arr[i]
        if hasattr(current_time, 'hour'):
            current_hour = current_time.hour
        else:
            current_hour = pd.to_datetime(current_time).hour
        
        if 9 <= current_hour < 15:
            trading_indices.append(i)
    
    # 优化后的循环
    for i in trading_indices:
        current_price = close_arr[i]
        current_time = datetime_arr[i]
        
        if position is not None:
            holding_bars = i - entry_idx
            if holding_bars > 30:
                profit = entry_price - current_price
                all_trades.append({
                    'date': date,
                    'entry_time': datetime_arr[entry_idx],
                    'exit_time': current_time,
                    'entry': entry_price,
                    'exit': current_price,
                    'profit': profit,
                    'exit_reason': '超时'
                })
                position = None
                continue
            
            if current_price <= entry_price - take_profit:
                all_trades.append({
                    'date': date,
                    'entry_time': datetime_arr[entry_idx],
                    'exit_time': current_time,
                    'entry': entry_price,
                    'exit': entry_price - take_profit,
                    'profit': take_profit,
                    'exit_reason': '止盈'
                })
                position = None
                continue
            elif current_price >= entry_price + stop_loss:
                all_trades.append({
                    'date': date,
                    'entry_time': datetime_arr[entry_idx],
                    'exit_time': current_time,
                    'entry': entry_price,
                    'exit': entry_price + stop_loss,
                    'profit': -stop_loss,
                    'exit_reason': '止损'
                })
                position = None
                continue
            continue
        
        # 布林带信号（向量化）
        if i >= bb_period + 1:
            prev_price = close_arr[i-1]
            current_bb_upper = bb_upper_arr[i]
            bb_sig = 'SELL' if (prev_price > current_bb_upper and current_price < current_bb_upper) else 'HOLD'
        else:
            bb_sig = 'HOLD'
        
        # RSI信号
        if i >= 30:
            rsi6 = rsi6_arr[i]
            rsi6_prev = rsi6_arr[i-1]
            rsi_sig = 'SELL' if (rsi6_prev > rsi_sell_high and rsi6 < rsi6_prev and rsi6 > rsi_sell_low) else 'HOLD'
        else:
            rsi_sig = 'HOLD'
        
        if bb_sig == 'SELL' and rsi_sig == 'SELL':
            # 过滤条件
            if pd.notna(recent_up_arr[i]) and recent_up_arr[i] >= filter_threshold:
                continue  # 过滤掉
            
            position = 'SELL'
            entry_price = current_price
            entry_idx = i
    
    return all_trades


def test_strategy_parallel(df, rsi_sell_high=90, rsi_sell_low=60,
                           take_profit=8, stop_loss=5, filter_threshold=7,
                           n_jobs=None):
    """测试策略（多进程并行版本）"""
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # 保留一个核心
    
    all_dates = sorted(df['date'].unique())
    
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
        test_strategy_single_day,
        rsi_sell_high=rsi_sell_high,
        rsi_sell_low=rsi_sell_low,
        take_profit=take_profit,
        stop_loss=stop_loss,
        filter_threshold=filter_threshold
    )
    
    # 并行处理
    with Pool(processes=n_jobs) as pool:
        results = pool.starmap(test_func, date_data)
    
    # 合并结果
    all_trades = []
    for trades in results:
        all_trades.extend(trades)
    
    elapsed_time = time.time() - start_time
    print(f"✅ 并行回测完成，耗时 {elapsed_time:.2f} 秒，平均每个交易日 {elapsed_time/len(date_data):.3f} 秒")
    
    return all_trades


def test_strategy(df, rsi_sell_high=90, rsi_sell_low=60,
                 take_profit=8, stop_loss=5,
                 filter_threshold=7, parallel=True, n_jobs=None):
    """测试策略（支持并行和串行模式）"""
    if parallel:
        return test_strategy_parallel(df, rsi_sell_high, rsi_sell_low,
                                     take_profit, stop_loss, filter_threshold, n_jobs)
    else:
        # 串行版本（保持兼容性）
        all_dates = sorted(df['date'].unique())
        all_trades = []
        
        for date in all_dates:
            df_day = df[df['date'] == date].reset_index(drop=True)
            trades = test_strategy_single_day(df_day, date, rsi_sell_high, rsi_sell_low,
                                            take_profit, stop_loss, filter_threshold)
            all_trades.extend(trades)
        
        return all_trades


def analyze_strategy_performance(trades):
    """分析策略表现"""
    if len(trades) == 0:
        print("⚠️  没有交易记录")
        return
    
    win_trades = [t for t in trades if t['profit'] > 0]
    loss_trades = [t for t in trades if t['profit'] < 0]
    
    total_profit = sum([t['profit'] for t in trades])
    win_rate = len(win_trades) / len(trades) * 100
    avg_profit = total_profit / len(trades)
    
    # 按日期统计
    daily_stats = {}
    for trade in trades:
        date = trade['date']
        if date not in daily_stats:
            daily_stats[date] = {'trades': [], 'profit': 0}
        daily_stats[date]['trades'].append(trade)
        daily_stats[date]['profit'] += trade['profit']
    
    profit_days = sum(1 for p in daily_stats.values() if p['profit'] > 0)
    
    print("\n" + "=" * 120)
    print("📊 策略盈利分析")
    print("=" * 120)
    
    print(f"\n总体表现:")
    print(f"   总交易: {len(trades)}笔")
    print(f"   盈利: {len(win_trades)}笔, 亏损: {len(loss_trades)}笔")
    print(f"   胜率: {win_rate:.2f}%")
    print(f"   总收益: {total_profit:+.2f}点")
    print(f"   平均每笔: {avg_profit:+.2f}点")
    print(f"   盈利天数: {profit_days}/{len(daily_stats)}天 ({profit_days/len(daily_stats)*100:.1f}%)")
    
    if win_trades:
        avg_win = sum([t['profit'] for t in win_trades]) / len(win_trades)
        max_win = max([t['profit'] for t in win_trades])
        print(f"\n盈利交易:")
        print(f"   平均盈利: {avg_win:.2f}点")
        print(f"   最大盈利: {max_win:.2f}点")
    
    if loss_trades:
        avg_loss = sum([t['profit'] for t in loss_trades]) / len(loss_trades)
        max_loss = min([t['profit'] for t in loss_trades])
        print(f"\n亏损交易:")
        print(f"   平均亏损: {avg_loss:.2f}点")
        print(f"   最大亏损: {max_loss:.2f}点")
    
    # 按退出原因统计
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    print(f"\n退出原因统计:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(trades) * 100
        print(f"   {reason}: {count}次 ({pct:.1f}%)")
    
    # 每日明细
    print(f"\n每日明细（前10天）:")
    sorted_dates = sorted(daily_stats.keys())
    for date in sorted_dates[:10]:
        stats = daily_stats[date]
        win = len([t for t in stats['trades'] if t['profit'] > 0])
        loss = len([t for t in stats['trades'] if t['profit'] < 0])
        status = "✅" if stats['profit'] > 0 else ("⚠️" if stats['profit'] == 0 else "❌")
        print(f"   {status} {date}: {len(stats['trades'])}笔, {win}盈/{loss}亏, {stats['profit']:+.2f}点")


def main():
    """主函数"""
    print("=" * 120)
    print("📊 JQData数据获取与策略分析")
    print("=" * 120)
    
    # 检查JQData
    if not JQDATA_AVAILABLE:
        print("\n❌ 请先安装jqdatasdk: pip install jqdatasdk")
        return
    
    # 使用配置的JQData账号信息
    global JQ_USERNAME, JQ_PASSWORD
    print(f"\n📝 使用JQData账号: {JQ_USERNAME}")
    
    # 登录JQData
    if not login_jqdata(JQ_USERNAME, JQ_PASSWORD):
        return
    
    # 创建数据表
    if not create_table_if_not_exists():
        return
    
    # 计算日期范围（试用账号：前15个月到前3个月）
    # 根据错误信息，账号权限范围是 2024-07-27 至 2025-08-03
    # 直接使用权限范围的最大值
    start_date = '2024-07-27'
    end_date = '2025-08-03'
    
    print(f"\n📅 数据范围: {start_date} 至 {end_date}")
    print(f"   (使用账号权限范围内的最大日期范围)")
    
    # 检查数据库中是否已有数据
    print("\n🔍 检查数据库中的现有数据...")
    existing_data = load_data_from_database('RB0', start_date, end_date)
    
    if existing_data is not None and len(existing_data) > 0:
        print(f"✅ 数据库已有 {len(existing_data)} 条数据")
        print("   使用现有数据进行策略分析...")
        df = existing_data
    else:
        # 获取新数据
        print("\n📥 正在从JQData获取数据...")
        df = get_futures_data_from_jqdata('RB0', start_date, end_date)
        if df is not None:
            print("\n💾 正在保存数据到数据库...")
            save_data_to_database(df, 'RB0')
    
    # 从数据库加载数据进行分析
    print("\n📊 从数据库加载数据进行分析...")
    df = load_data_from_database('RB0', start_date, end_date)
    
    if df is None or len(df) == 0:
        print("❌ 没有可用数据进行分析")
        return
    
    # 测试策略
    print("\n" + "=" * 120)
    print("📈 策略回测")
    print("=" * 120)
    
    print("\n策略参数:")
    print("   RSI参数: >90→>60（严格）")
    print("   过滤条件: 最近10根上涨数 < 7")
    print("   止盈止损: +8/-5")
    
    # 检测CPU核心数
    available_cores = cpu_count()
    print(f"\n💻 系统CPU核心数: {available_cores}")
    print(f"   将使用 {max(1, available_cores - 1)} 个核心进行并行回测")
    
    start_time = time.time()
    trades = test_strategy(df, rsi_sell_high=90, rsi_sell_low=60,
                          take_profit=8, stop_loss=5, filter_threshold=7,
                          parallel=True)
    total_time = time.time() - start_time
    print(f"\n⏱️  总回测时间: {total_time:.2f} 秒")
    
    # 分析策略表现
    analyze_strategy_performance(trades)
    
    print("\n" + "=" * 120)
    print("✅ 分析完成")
    print("=" * 120)


if __name__ == '__main__':
    main()

