import pandas as pd
import numpy as np
import os

def run_backtest(df, adx_length, threshold):
    # Copy df to avoid side effects
    df = df.copy()
    
    # 1. Indicators
    # TR
    c = df['Close']
    h = df['High']
    l = df['Low']
    c_prev = c.shift(1)
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # DM
    h_prev = h.shift(1)
    l_prev = l.shift(1)
    up_move = h - h_prev
    down_move = l_prev - l
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth (Wilder)
    def smooth(series, length):
        # Concise pandas ewm
        return series.ewm(alpha=1/length, adjust=False).mean()
        # Wait, standard Wilder is alpha=1/N. Pandas ewm with adjust=False matches?
        # y[t] = (1-a)y[t-1] + a*x[t].
        # Wilder: S[t] = S[t-1] - S[t-1]/N + X[t] = S[t-1](1-1/N) + X[t].
        # COEFF MATCH for S[t-1].
        # But Wilder has '1' for X[t], pandas has 'a' (1/N).
        # So pandas result is exactly "Wilder / N".
        # Since we divide DI+/TR, the factor 1/N cancels out. 
        # So yes, ewm works for RATIOS.
    
    # ACTUALLY my manual loop used "current_sum = prev - prev/N + val".
    # This matches Wilder exactly.
    # Pandas EWM matches Exponential Moving Average.
    # EMA(X) ~ Wilder(X) / N?
    # Let's stick to the manual loop to be consistent with main script.
    
    def smooth_wilder_sum(series, length):
        res = np.zeros_like(series)
        # Handle nan start
        first_valid = series.first_valid_index()
        if first_valid is None: return pd.Series(res, index=series.index)
        
        start_i = series.index.get_loc(first_valid)
        vals = series.values
        out = np.full_like(vals, np.nan)
        
        current_sum = vals[start_i]
        out[start_i] = current_sum
        alpha = 1.0 / length
        
        for i in range(start_i + 1, len(vals)):
            val = vals[i] if not np.isnan(vals[i]) else 0
            prev = out[i-1]
            if np.isnan(prev):
                current_sum = val
            else:
                current_sum = prev - (prev * alpha) + val
            out[i] = current_sum
        return pd.Series(out, index=series.index)

    smoothed_tr = smooth_wilder_sum(true_range, adx_length)
    smoothed_dm_plus = smooth_wilder_sum(pd.Series(dm_plus, index=df.index), adx_length)
    smoothed_dm_minus = smooth_wilder_sum(pd.Series(dm_minus, index=df.index), adx_length)
    
    di_plus = (smoothed_dm_plus / smoothed_tr) * 100
    di_minus = (smoothed_dm_minus / smoothed_tr) * 100
    
    # Trend Score
    # Simplified vectorized approach approx mimicking the loop?
    # No, the loop has memory ("curr_pos += 1"). Hard to vectorize perfectly.
    # Loop is fast enough.
    
    pos_counts = []
    neg_counts = []
    curr_pos = 0
    curr_neg = 0
    
    dp_vals = di_plus.values
    dm_vals = di_minus.values
    
    for i in range(len(df)):
        if i == 0 or np.isnan(dp_vals[i]) or np.isnan(dp_vals[i-1]):
            pos_counts.append(0)
            neg_counts.append(0)
            continue
            
        dp = dp_vals[i]
        dp_prev = dp_vals[i-1]
        dm = dm_vals[i]
        dm_prev = dm_vals[i-1]
        
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
    
    # 2. Strategy
    df['in_range'] = (df.index >= '2018-01-01') & (df.index <= '2069-12-31')
    
    if threshold == 0:
        df['signal'] = np.where((df['trend_score'] > 0) & df['in_range'], 1, 0)
    else:
         df['signal'] = np.where((df['trend_score'] >= threshold) & df['in_range'], 1, 0)
         
    # Backtest Loop
    # Simplified logic: 
    # Use shifted signal for position
    df['pos'] = df['signal'].shift(1).fillna(0)
    
    # Vectorized Returns
    # Returns = (Close / Close_prev) - 1.
    # Strat Returns = Returns * Pos.
    # Deduct transaction costs?
    # Pos change: abs(Pos - Pos_prev).
    
    df['pct_change'] = df['Close'].pct_change()
    df['strat_ret'] = df['pct_change'] * df['pos']
    
    # Commission
    commission = 0.001
    df['pos_change'] = (df['pos'] - df['pos'].shift(1)).abs()
    cost = df['pos_change'] * commission
    
    df['net_ret'] = df['strat_ret'] - cost
    
    # Equity
    # (1+r).cumprod()
    df['equity'] = 10000 * (1 + df['net_ret']).cumprod()
    
    final_eq = df['equity'].iloc[-1]
    trades = df['pos_change'].sum() / 2 # Buy+Sell=2
    
    # Max DD
    roll_max = df['equity'].cummax()
    dd = (df['equity'] - roll_max) / roll_max
    max_dd = dd.min() * 100
    
    return final_eq, trades, max_dd

# Load Data
df = pd.read_csv('data/ethusd.csv', parse_dates=['Date'], index_col='Date').sort_index()

print(f"{'ADX':<5} | {'Thresh':<6} | {'Equity':<12} | {'Trades':<8} | {'MaxDD':<8}")
print("-" * 50)

for adx in range(9, 15):
    for thresh in [0, 2]:
        eq, tr, dd = run_backtest(df, adx, thresh)
        print(f"{adx:<5} | {thresh:<6} | ${eq:<11,.0f} | {tr:<8.1f} | {dd:.2f}%")
