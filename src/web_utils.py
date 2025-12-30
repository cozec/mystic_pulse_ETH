import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta

def smooth_wilder_sum(series, length):
    """Wilder's smoothing recursively."""
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        return pd.Series(np.full_like(series.values, np.nan), index=series.index)
    
    vals = series.values
    out = np.full_like(vals, np.nan)
    start_i = series.index.get_loc(first_valid_idx)
    
    current_sum = vals[start_i]
    out[start_i] = current_sum
    alpha = 1.0 / length
    
    for i in range(start_i + 1, len(vals)):
        val = vals[i]
        prev = out[i-1]
        if np.isnan(prev):
            current_sum = val
        else:
            current_sum = prev - (prev * alpha) + val
        out[i] = current_sum
    
    return pd.Series(out, index=series.index)

def calculate_indicators(df):
    """
    Calculates Mystic Pulse indicators (Trend Score, Signals)
    """
    # Parameters
    adx_length = 9
    
    # OHLC
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']

    # True Range
    c_prev = c.shift(1)
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # DM+ and DM-
    h_prev = h.shift(1)
    l_prev = l.shift(1)
    up_move = h - h_prev
    down_move = l_prev - l

    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    dm_plus = pd.Series(dm_plus, index=df.index)
    dm_minus = pd.Series(dm_minus, index=df.index)

    # Wilder's Smoothing
    smoothed_tr = smooth_wilder_sum(true_range, adx_length)
    smoothed_dm_plus = smooth_wilder_sum(dm_plus, adx_length)
    smoothed_dm_minus = smooth_wilder_sum(dm_minus, adx_length)

    di_plus = (smoothed_dm_plus / smoothed_tr) * 100
    di_minus = (smoothed_dm_minus / smoothed_tr) * 100

    # Trend Counting Logic
    pos_counts = []
    neg_counts = []
    curr_pos = 0
    curr_neg = 0

    di_p_vals = di_plus.values
    di_m_vals = di_minus.values

    for i in range(len(df)):
        if i == 0:
            pos_counts.append(0)
            neg_counts.append(0)
            continue
        
        dp = di_p_vals[i]
        dp_prev = di_p_vals[i-1]
        dm = di_m_vals[i]
        dm_prev = di_m_vals[i-1]
        
        if np.isnan(dp) or np.isnan(dp_prev) or np.isnan(dm) or np.isnan(dm_prev):
            pos_counts.append(curr_pos)
            neg_counts.append(curr_neg)
            continue
        
        is_bull = (dp > dp_prev) and (dp > dm)
        is_bear = (dm > dm_prev) and (dm > dp)
        
        if is_bull:
            curr_pos += 1
            curr_neg = 0
        
        if is_bear:
            curr_neg += 1
            curr_pos = 0
        
        pos_counts.append(curr_pos)
        neg_counts.append(curr_neg)

    df['trend_score'] = np.array(pos_counts) - np.array(neg_counts)
    
    # Logic: Score > 0 is Long (1), <= 0 is Flat (0)
    df['signal'] = np.where(df['trend_score'] > 0, 1, 0)
    
    # Identify Entry/Exit points
    df['signal_change'] = df['signal'].diff()
    # 1 = Entry (0 -> 1)
    # -1 = Exit (1 -> 0)
    
    return df

def get_eth_data_with_live():
    """
    Loads historical CSV and appends latest live data from yfinance
    """
    csv_path = os.path.join("data", "ethusd.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
        
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    
    # Fetch live data
    try:
        ticker = yf.Ticker("ETH-USD")
        live_df = ticker.history(period="5d")
        
        if not live_df.empty:
            live_df = live_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            live_df.index = live_df.index.tz_localize(None)
            
            latest_date = live_df.index[-1]
            
            if latest_date not in df.index:
                # Append
                df = pd.concat([df, live_df.tail(1)])
            else:
                # Update
                df.loc[latest_date] = live_df.iloc[-1]
            
            df = df.sort_index()
            
    except Exception as e:
        print(f"Web Utils: Error fetching live data: {e}")
        
    return df
