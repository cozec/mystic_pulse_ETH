import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import os
import yfinance as yf # Added for live data

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

# Load ETH data
data_path = os.path.join("data", "ethusd.csv")
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
df = df.sort_index()

# --- FETCH LIVE DATA ---
print("Fetching live data for today...")
try:
    ticker = yf.Ticker("ETH-USD")
    # Fetch 5 days to be safe, but we only want the latest
    live_df = ticker.history(period="5d")
    
    if not live_df.empty:
        # Standardize columns
        live_df = live_df[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Ensure index is timezone-naive to match CSV
        live_df.index = live_df.index.tz_localize(None)
        
        # Get the very latest date
        latest_date = live_df.index[-1]
        
        # Check if this date is already in our CSV df
        if latest_date not in df.index:
            print(f"Appending new data for {latest_date} (Price: {live_df['Close'].iloc[-1]})")
            df = pd.concat([df, live_df.tail(1)])
        else:
            print(f"Updating data for {latest_date} (Price: {live_df['Close'].iloc[-1]})")
            df.loc[latest_date] = live_df.iloc[-1]
            
        df = df.sort_index()
        
except Exception as e:
    print(f"Error fetching live data: {e}")
# -----------------------

# Filter for past 6 months

# Filter for past 6 months
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

df_ytd = df[df.index >= start_date].copy()

if df_ytd.empty:
    print("No data available for past 6 months")
    exit()

# Calculate Mystic Pulse indicator
adx_length = 9
smoothing_factor = 1

# OHLC (no smoothing since factor=1)
o, h, l, c = df_ytd['Open'], df_ytd['High'], df_ytd['Low'], df_ytd['Close']

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

dm_plus = pd.Series(dm_plus, index=df_ytd.index)
dm_minus = pd.Series(dm_minus, index=df_ytd.index)

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

for i in range(len(df_ytd)):
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

df_ytd['trend_score'] = np.array(pos_counts) - np.array(neg_counts)

# Detect Entry/Exit signals
df_ytd['signal'] = np.where(df_ytd['trend_score'] > 0, 1, 0)
df_ytd['signal_change'] = df_ytd['signal'].diff()

entries = df_ytd[df_ytd['signal_change'] == 1].copy()
exits = df_ytd[df_ytd['signal_change'] == -1].copy()

# Create Plot with 3 panels
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[5, 2, 0.3], hspace=0.05)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)

# Panel 1: Candlestick Chart
for idx, row in df_ytd.iterrows():
    date = mdates.date2num(idx)
    open_p = row['Open']
    close_p = row['Close']
    high_p = row['High']
    low_p = row['Low']
    
    color = '#26a69a' if close_p >= open_p else '#ef5350'
    
    # High-Low line
    ax1.plot([date, date], [low_p, high_p], color=color, linewidth=1)
    
    # Body
    height = abs(close_p - open_p)
    bottom = min(open_p, close_p)
    rect = Rectangle((date - 0.3, bottom), 0.6, height, facecolor=color, edgecolor=color)
    ax1.add_patch(rect)

# Add baseline (starting price)
baseline = df_ytd['Close'].iloc[0]
ax1.axhline(baseline, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Entry markers (Blue up arrows)
for idx, row in entries.iterrows():
    date = mdates.date2num(idx)
    price = row['Low']
    ax1.plot(date, price, marker='^', markersize=10, color='blue', zorder=5)
    ax1.plot(date, price, 'o', markersize=8, color='white', markeredgecolor='blue', markeredgewidth=2, zorder=6)
    
    # Label
    ax1.annotate(f'Bull Start\n+{price:.4f}', 
                xy=(date, price), 
                xytext=(0, -30),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.8))

# Exit markers (Purple down arrows)
for idx, row in exits.iterrows():
    date = mdates.date2num(idx)
    price = row['High']
    ax1.plot(date, price, marker='v', markersize=10, color='purple', zorder=5)
    ax1.plot(date, price, 'o', markersize=8, color='white', markeredgecolor='purple', markeredgewidth=2, zorder=6)
    
    # Label
    ax1.annotate(f'Trend Break\n-{price:.4f}', 
                xy=(date, price), 
                xytext=(0, 30),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple', alpha=0.8))

ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.2, linestyle=':')
ax1.xaxis.set_visible(False)
ax1.set_title('ETH-USD Past 6M - Mystic Pulse Strategy', fontsize=14, fontweight='bold', pad=15)

# Panel 2: Trend Score Indicator (Bar Chart)
dates_num = mdates.date2num(df_ytd.index)

# Create gradient colors based on trend_score magnitude
# Dark colors for low magnitude, bright/neon for high magnitude
colors = []
max_abs_score = max(abs(df_ytd['trend_score'].min()), abs(df_ytd['trend_score'].max()))

for score in df_ytd['trend_score']:
    if score > 0:
        # Positive: from dark green (#005A00) to neon green (#00FF66)
        # Normalize score to 0-1 based on max
        intensity = min(abs(score) / max(10, max_abs_score), 1.0)
        # Interpolate between dark and neon
        r = int(0x00 + (0x00 - 0x00) * intensity)
        g = int(0x5A + (0xFF - 0x5A) * intensity)
        b = int(0x00 + (0x66 - 0x00) * intensity)
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    elif score < 0:
        # Negative: from dark red (#7A0000) to neon red (#FF1A1A)
        intensity = min(abs(score) / max(10, max_abs_score), 1.0)
        r = int(0x7A + (0xFF - 0x7A) * intensity)
        g = int(0x00 + (0x1A - 0x00) * intensity)
        b = int(0x00 + (0x1A - 0x00) * intensity)
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    else:
        colors.append('#808080')  # Gray for neutral

ax2.bar(dates_num, df_ytd['trend_score'].abs(), color=colors, width=0.8)
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.set_ylabel('Trend Score (Abs)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2, linestyle=':')
ax2.xaxis.set_visible(False)

# Panel 3: Position Indicator (Color Band)
ax3.set_ylim(0, 1)
ax3.set_yticks([])
ax3.set_ylabel('')

# Draw colored rectangles for position
for i in range(len(df_ytd) - 1):
    date_start = mdates.date2num(df_ytd.index[i])
    date_end = mdates.date2num(df_ytd.index[i + 1])
    color = '#26a69a' if df_ytd['signal'].iloc[i] == 1 else '#ef5350'
    rect = Rectangle((date_start, 0), date_end - date_start, 1, facecolor=color, edgecolor='none')
    ax3.add_patch(rect)

# Last bar
date_start = mdates.date2num(df_ytd.index[-1])
date_end = date_start + 1
color = '#26a69a' if df_ytd['signal'].iloc[-1] == 1 else '#ef5350'
rect = Rectangle((date_start, 0), date_end - date_start, 1, facecolor=color, edgecolor='none')
ax3.add_patch(rect)

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax3.set_xlabel('Date', fontsize=12, fontweight='bold')

plt.tight_layout()

output_path = os.path.join("plots", "eth_6m_mystic_pulse.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {output_path}")
print(f"Period: {df_ytd.index[0].date()} to {df_ytd.index[-1].date()}")
print(f"Current Trend Score: {df_ytd['trend_score'].iloc[-1]}")
print(f"Current Signal: {'LONG' if df_ytd['trend_score'].iloc[-1] > 0 else 'FLAT'}")
print(f"Entry Signals: {len(entries)}")
print(f"Exit Signals: {len(exits)}")
