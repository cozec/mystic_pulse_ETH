import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_strategy(ticker="BTC-USD", csv_filename="btcusd.csv"):
    print(f"Running Mystic Pulse Strategy for {ticker}...")
    # 1. Load Data
    data_path = os.path.join("data", csv_filename)
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()

    # Filter Date Range: 2018-01-01 to 2069-12-31
    # Note: User request says "2018-2069", but I'll ensure we start from 2018.
    # However, for correct smoothing initialization, it might be good to have some pre-2018 data 
    # and then cut the results. But "Avoid forward looking" means we can't use future data. 
    # We can use pre-2018 data for warmup. 
    # The script says "Date Filter: 2018-2069" for STRATEGY EXECUTION.
    # Meaning indicators can calculate before, but trades only start 2018.
    
    # Let's keep all data for calc, then filter for signals.
    
    # 2. Inputs
    adx_length = 9
    smoothing_factor = 1 # OHLC SMA length. 1 means no smoothing.
    
    # 3. Core Calculations
    # OHLC Smoothing (SMA 1 = Raw values)
    # If smoothing_factor > 1, use rolling mean.
    if smoothing_factor > 1:
        o = df['Open'].rolling(window=smoothing_factor).mean()
        h = df['High'].rolling(window=smoothing_factor).mean()
        l = df['Low'].rolling(window=smoothing_factor).mean()
        c = df['Close'].rolling(window=smoothing_factor).mean()
    else:
        o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    
    # True Range
    # TR = max(H-L, abs(H-Cprev), abs(L-Cprev))
    c_prev = c.shift(1)
    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # DM+ and DM-
    # dm_plus = (high - prev_high > prev_low - low) ? max(high - prev_high, 0) : 0
    h_prev = h.shift(1)
    l_prev = l.shift(1)
    
    up_move = h - h_prev
    down_move = l_prev - l
    
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    dm_plus = pd.Series(dm_plus, index=df.index)
    dm_minus = pd.Series(dm_minus, index=df.index)
    
    # 4. Wilder's Smoothing (Recursive)
    # Formula: S[i] = S[i-1] - (S[i-1]/N) + X[i]
    # S[i] = S[i-1] * (1 - 1/N) + X[i]
    # This is equivalent to pandas ewm with alpha=1/N using adjust=False?
    # Pandas ewm formula: y[i] = (1-alpha)*y[i-1] + alpha*x[i]
    # Here we have:       S[i] = (1 - 1/N)*S[i-1] + 1*X[i]
    # The coefficient for X[i] is 1, not alpha.
    # So Pandas ewm result needs to be divided by alpha? No.
    # Let's just implement the loop to be precise and safe, as python loop over 4k rows is fast.
    
    def smooth_wilder_sum(series, length):
        # Initialize with first value? Or 0?
        # Pine: na(s[1]) ? x : ...
        # So first value is x.
        res = np.zeros_like(series)
        res[0] = series.iloc[0] # Handle NaN? series.iloc[0] might be nan if calc involves shift.
        
        # We need to handle initial NaNs (e.g. from shift).
        # Find first valid index
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            return pd.Series(res, index=series.index)
            
        # We'll use a python list for speed
        vals = series.values
        out = np.full_like(vals, np.nan)
        
        # Start from first valid
        # Find integer index of first valid
        start_i = series.index.get_loc(first_valid_idx)
        
        current_sum = vals[start_i]
        out[start_i] = current_sum
        
        alpha = 1.0 / length
        
        for i in range(start_i + 1, len(vals)):
            val = vals[i]
            if np.isnan(val):
                # If input is nan, keep previous? Or nan?
                # Usually propagate nan or skip. Pine propagates text usually?
                # In Pine, if x is NaN, the formula might result in NaN.
                # Let's assume input isn't NaN after the start.
                val = 0 # Treat as 0? No, that breaks logic.
            
            # recursive: S[i] = S[i-1] - (S[i-1]/N) + X[i]
            prev = out[i-1]
            if np.isnan(prev):
                current_sum = val
            else:
                current_sum = prev - (prev * alpha) + val
            out[i] = current_sum
            
        return pd.Series(out, index=series.index)

    smoothed_tr = smooth_wilder_sum(true_range, adx_length)
    smoothed_dm_plus = smooth_wilder_sum(dm_plus, adx_length)
    smoothed_dm_minus = smooth_wilder_sum(dm_minus, adx_length)
    
    di_plus = (smoothed_dm_plus / smoothed_tr) * 100
    di_minus = (smoothed_dm_minus / smoothed_tr) * 100
    
    # 5. Trend Counting Logic
    # var int positive_count = 0
    # var int negative_count = 0
    # if (cond_plus) pos++, neg=0
    # if (cond_minus) neg++, pos=0
    
    pos_counts = []
    neg_counts = []
    
    curr_pos = 0
    curr_neg = 0
    
    # Pre-calculate conditions to speed up loop?
    # but they depend on previous values of each other? No, DI values are fixed.
    # cond_plus: di_plus > di_plus[1] and di_plus > di_minus
    # cond_minus: di_minus > di_minus[1] and di_minus > di_plus
    
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
        
        # Check NaNs
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
    
    # 6. Strategy Calculation
    # Logic:
    # Buy (Long) when trend_score > 0 and inDateRange.
    # Sel (Exit) when trend_score <= 0.
    
    df['in_range'] = (df.index >= '2018-01-01') & (df.index <= '2069-12-31')
    
    # Signal: 1 for Hold, 0 for Cash
    # We enter when score > 0. We stay until score <= 0.
    # So if score > 0, we want to be Long.
    # If score <= 0, we want to be flat.
    # Wait, check Pine:
    # if (longCondition): entry
    # if (exitCondition): close
    # longCondition = score > 0.
    # exitCondition = score <= 0.
    # So yes, Position = 1 if score > 0, else 0.
    
    df['signal'] = np.where((df['trend_score'] > 0) & df['in_range'], 1, 0)
    
    # Calculate Returns
    # 100% Equity.
    # Commission 0.1% per trade.
    # Slippage 1 tick?
    # Let's do a vector backtest with costs.
    
    # Position change
    # position[t] determines if we hold return of t+1?
    # Usually: We calculate signal at Close[t]. We trade at Open[t+1] or Close[t]?
    # Pine Strategies usually trade at Open of next bar unless `calc_on_every_tick` or specific settings.
    # Default: "After header... orders are filled on the next available price... default is Open of next bar."
    # So signal[t] triggers trade at Open[t+1].
    
    # Position vector
    df['position'] = df['signal'].shift(1).fillna(0) # Position held during day t
    
    # Daily Returns
    df['pct_change'] = df['Close'].pct_change()
    
    # Strategy Gross Returns
    df['strat_ret'] = df['position'] * df['pct_change']
    
    # Transaction Costs
    # Trade occurs when position changes.
    # Change = position[t] - position[t-1]
    # Cost = abs(change) * (commission + slippage_pct)
    
    # Commission: 0.1% = 0.001
    commission = 0.001
    
    # Slippage: "1 tick".
    # Since we are doing % returns, we need slippage in %.
    # 1 tick / Price.
    # This is variable.
    # Let's Approximate or calc exact.
    # Price at execution. If moving from 0->1, we buy at Open[t].
    # Slippage is cost relative to Open[t].
    # Cost = 1 tick / Open[t].
    # Tick size for BTC = 0.01 usually.
    tick_size = 0.01
    
    # Trades happen at Open[t] based on Signal[t-1].
    # So we use Open[t] for cost calculation.
    
    trades = df['position'].diff() != 0 
    # Note: df['position'] is already shifted. so position[t] is what we hold at t.
    # If position[t] != position[t-1], we traded at Open[t].
    
    # We need to handle the start carefully.
    
    df['txn_cost'] = 0.0
    
    # Vectorized cost
    # Change at t means trade at t.
    changes = df['position'].diff().abs()
    # Cost = change * (comm + tick/Open) * 1 (since 100% equity? rough approx)
    # Actually, if we buy 1 unit of equity worth.
    # Fee is 0.1% of value.
    # Slippage is fixed amount per unit? No, "1 tick slippage" usually means price is filled at PRICE +/- tick.
    # So effective price is worse by 1 tick.
    # Percentage impact is tick/Price.
    
    # Handle NaN in Open
    opens = df['Open']
    
    cost_series = changes * (commission + (tick_size / opens))
    df['txn_cost'] = cost_series.fillna(0)
    
    df['strat_ret_net'] = df['strat_ret'] - df['txn_cost']
    
    # Verify: If position went 0->1 (Buy), we pay cost.
    # Return for that day is Open->Close (if we bought at Open).
    # Wait, `df['pct_change']` is (Close[t] - Close[t-1]) / Close[t-1].
    # If we enter at Open[t], our return for day t is (Close[t] - Open[t]) / Open[t] ?
    # AND we missed the gap (Open[t] - Close[t-1]).
    # Standard vector backtest `pos.shift(1) * pct_change` implies we captured the Close-to-Close move.
    # This implies we bought at Close[t-1].
    # But Pine trades at Open[t].
    # If we buy at Open[t], we DO NOT capture (Open[t] - Close[t-1]).
    # So logic:
    # If pos[t] == 1 (Holding):
    #   If pos[t-1] == 1: Held overnight. Return = (C[t]-C[t-1])/C[t-1].
    #   If pos[t-1] == 0: Just Bought at Open[t]. Return = (C[t]-O[t])/O[t].
    # If pos[t] == 0 (Cash):
    #   If pos[t-1] == 1: Just Sold at Open[t]. Return = (O[t]-C[t-1])/C[t-1]. (Gap)
    #   If pos[t-1] == 0: Flat. Return = 0.
    
    # Let's refine the return calc to match "Trade at Open".
    
    df['ret_close_close'] = df['Close'].pct_change() # (C[t] - C[t-1])/C[t-1]
    df['ret_open_close']  = (df['Close'] - df['Open']) / df['Open']
    df['ret_close_open']  = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # We need a loop or complex vector logic. Loop is cleaner for logic.
    equity = [10000.0] # Start
    position = 0 # 0 or 1
    
    # Signals are calculated at end of day t (using trend_score[t]).
    # Executed at Open[t+1].
    
    # We iterate t from 2018.
    df_sim = df[df.index >= '2018-01-01'].copy()
    
    # We need the trend_score from the PREVIOUS day to decide action at OPEN of CURRENT day.
    # `signal` column represents "Desired Position" based on Close[t].
    # `df_sim` has `signal` computed.
    # row i: Date i.
    # At Open[i], we look at Signal[i-1].
    
    # Re-map signals to avoid confusion
    # previous_signal[i] = signal[i-1] (from the full df)
    # Be careful with index alignment.
    
    # Let's just user iterrows() on filtered df.
    # We need 'signal' of prev row.
    
    # Add 'prev_signal' col
    df['prev_signal'] = df['signal'].shift(1)
    df_sim = df[df.index >= '2018-01-01'].copy()
    
    curr_equity = 10000.0
    curr_pos = 0 # 0=Cash, 1=Invested
    equity_curve = []
    
    # For stats
    trades_list = []
    entry_price = 0.0
    entry_date = None
    
    for i in range(len(df_sim)):
        date = df_sim.index[i]
        row = df_sim.iloc[i]
        
        # Desired position
        target_pos = row['prev_signal']
        if np.isnan(target_pos): target_pos = 0
        
        open_p = row['Open']
        close_p = row['Close']
        
        # Trade Execution at Open
        if target_pos == 1 and curr_pos == 0:
            # BUY
            effective_price = open_p + tick_size
            shares = curr_equity / effective_price
            cost = curr_equity * commission
            curr_equity -= cost
            
            curr_pos = 1
            entry_price = effective_price
            entry_date = date
            
            # Record trade
            trades_list.append({'Date': date, 'Type': 'Buy', 'Price': effective_price, 'Equity': curr_equity})
            
        elif target_pos == 0 and curr_pos == 1:
            # SELL
            effective_price = open_p - tick_size
            val_pre_cost = shares * effective_price
            cost = val_pre_cost * commission
            curr_equity = val_pre_cost - cost
            
            curr_pos = 0
            
            trades_list.append({'Date': date, 'Type': 'Sell', 'Price': effective_price, 'Equity': curr_equity})
        
        # Intraday PnL (Open to Close)
        if curr_pos == 1:
            # We are holding shares. Value updates from Effective Open to Close.
            # If we just bought, base is effective_price.
            # If we held, base is (Open? No).
            # Simplest: Update Equity to Close Value.
            # If curr_pos = 1, Equity = shares * Close
            curr_equity = shares * close_p
        
        equity_curve.append(curr_equity)
        
    df_sim['Equity'] = equity_curve
    
    # Metrics
    if not equity_curve:
        print("No data in range.")
        return

    # Helper function for metrics
    def calculate_metrics(series, capital=10000.0):
        final_val = series.iloc[-1]
        total_ret = (final_val - capital) / capital * 100
        
        # CAGR
        n_days = (series.index[-1] - series.index[0]).days
        cagr = ((final_val / capital) ** (365 / n_days) - 1) * 100 if n_days > 0 else 0
        
        # Max Drawdown
        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max * 100
        max_dd = drawdown.min()
        
        # Daily Returns for Risk Metrics
        daily_ret = series.pct_change().dropna()
        
        # Risk (Annualized Volatility)
        volatility = daily_ret.std() * np.sqrt(365) * 100
        
        # Sharpe (assuming Rf=0)
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() != 0 else 0
        
        # Sortino (assuming Rf=0)
        downside_ret = daily_ret[daily_ret < 0]
        sortino = (daily_ret.mean() / downside_ret.std()) * np.sqrt(365) if downside_ret.std() != 0 else 0
        
        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            "Final Equity": final_val,
            "Total Return": total_ret,
            "CAGR": cagr,
            "Max Drawdown": max_dd,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar
        }

    # Strategy Metrics
    df_sim['Equity'] = equity_curve # Ensure series
    strat_metrics = calculate_metrics(pd.Series(equity_curve, index=df_sim.index), 10000)
    
    # Buy & Hold Metrics
    start_price = df_sim['Open'].iloc[0]
    bh_shares = 10000 / start_price
    df_sim['BuyHold'] = bh_shares * df_sim['Close']
    bh_metrics = calculate_metrics(df_sim['BuyHold'], 10000)
    
    print("-" * 60)
    print(f"{'Metric':<20} | {'Strategy':<15} | {'Buy & Hold':<15}")
    print("-" * 60)
    for k in strat_metrics:
        if k == "Final Equity":
            print(f"{k:<20} | ${strat_metrics[k]:<14,.2f} | ${bh_metrics[k]:<14,.2f}")
        elif "Ratio" in k:
            print(f"{k:<20} | {strat_metrics[k]:<15.2f} | {bh_metrics[k]:<15.2f}")
        else:
            print(f"{k:<20} | {strat_metrics[k]:<14.2f}% | {bh_metrics[k]:<14.2f}%")
    print("-" * 60)
    print(f"Total Trades: {len(trades_list)//2}")
    
    # Calculate Annual Returns
    print("\n" + "=" * 60)
    print(f"{'Year':<10} | {'Strategy Return':<20} | {'Buy & Hold Return':<20}")
    print("-" * 60)
    
    # Resample to Annual
    # Take last equity of each year
    annual_equity = df_sim['Equity'].resample('Y').last()
    annual_bh = df_sim['BuyHold'].resample('Y').last()
    
    # Add start capital for first year calculation (if not in resample)
    # Actually pct_change on resampled data works if we prepend start.
    # Or just loop keys.
    
    years = annual_equity.index.year.unique()
    
    # Previous year end equity
    prev_eq = 10000.0
    prev_bh = 10000.0
    
    for year_date in annual_equity.index:
        year = year_date.year
        curr_eq = annual_equity[year_date]
        curr_bh = annual_bh[year_date]
        
        # Log Return = ln(End/Start). User asked for "log return"?
        # "log return for each year". 
        # Usually user means %. Let's provide % and Log? 
        # "Log return" specifically means ln(P_t/P_{t-1}).
        # Let's verify if user means strict log return or just annual % return.
        # "log return" is specific. I will provide both or Log return.
        # Log Return = np.log(End / Start)
        
        strat_log_ret = np.log(curr_eq / prev_eq)
        bh_log_ret = np.log(curr_bh / prev_bh)
        
        # Percentage for display as well?
        strat_pct = (curr_eq - prev_eq) / prev_eq * 100
        bh_pct = (curr_bh - prev_bh) / prev_bh * 100
        
        print(f"{year:<10} | {strat_log_ret:<20.4f} | {bh_log_ret:<20.4f}")
        
        prev_eq = curr_eq
        prev_bh = curr_bh
        
    print("=" * 60)
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Equity Curve
    ax1.plot(df_sim.index, df_sim['Equity'], label='Strategy Equity', color='blue')
    
    # Buy Hold from start date
    start_price = df_sim['Open'].iloc[0]
    bh_shares = 10000 / start_price
    df_sim['BuyHold'] = bh_shares * df_sim['Close']
    ax1.plot(df_sim.index, df_sim['BuyHold'], label='Buy & Hold', color='gray', alpha=0.5, linestyle='--')
    
    ax1.set_title(f'Mystic Pulse Strategy Equity ({ticker})')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: BTC Price and Signals
    ax2.plot(df_sim.index, df_sim['Close'], label='BTC Price', color='black', alpha=0.6)
    
    # Extract Buy/Sell points for plotting
    buy_dates = [t['Date'] for t in trades_list if t['Type'] == 'Buy']
    buy_prices = [t['Price'] for t in trades_list if t['Type'] == 'Buy']
    
    sell_dates = [t['Date'] for t in trades_list if t['Type'] == 'Sell']
    sell_prices = [t['Price'] for t in trades_list if t['Type'] == 'Sell']
    
    ax2.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy', zorder=5)
    ax2.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell', zorder=5)
    
    ax2.set_title('BTC Price & Trade Signals')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log') # Log scale for BTC price is usually better over long term
    
    plt.tight_layout()
    
    output_plot = os.path.join("plots", f"mystic_pulse_{ticker}_backtest.png")
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # Save Results
    res_path = os.path.join("results", f"mystic_pulse_{ticker}_trades.csv")
    pd.DataFrame(trades_list).to_csv(res_path, index=False)
    print(f"Trade log saved to {res_path}")

    # --- Best/Worst Trades PDF Generation ---
    from matplotlib.backends.backend_pdf import PdfPages
    
    # 1. Reconstruct Round-Trip Trades
    round_trips = []
    current_entry = None
    
    for t in trades_list:
        if t['Type'] == 'Buy':
            current_entry = t
        elif t['Type'] == 'Sell' and current_entry is not None:
            # Calculate PnL
            # Return = (Exit Price - Entry Price) / Entry Price
            # Note: This ignores fees for simple "best trade" ranking, or we can use Equity change?
            # Using Equity change is more accurate for "Net result" but price move is better for "Market move".
            # User wants "best trades". Usually means % gain.
            pnl_pct = (t['Price'] - current_entry['Price']) / current_entry['Price'] * 100
            
            round_trips.append({
                'Entry Date': current_entry['Date'],
                'Exit Date': t['Date'],
                'Entry Price': current_entry['Price'],
                'Exit Price': t['Price'],
                'PnL %': pnl_pct
            })
            current_entry = None
            
    # Sort by PnL
    round_trips.sort(key=lambda x: x['PnL %'], reverse=True)
    
    best_10 = round_trips[:10]
    worst_10 = round_trips[-10:]
    
    pdf_path = os.path.join("plots", f"mystic_pulse_{ticker}_trade_analysis.pdf")
    
    print(f"Generating PDF report: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        # Title Page
        plt.figure(figsize=(11.69, 8.27)) # A4 landscape
        plt.text(0.5, 0.5, "Mystic Pulse Strategy\nTrade Analysis\n\nTop 10 Best & Worst Trades", 
                 horizontalalignment='center', verticalalignment='center', fontsize=24)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        trades_to_plot = [("Best", t) for t in best_10] + [("Worst", t) for t in worst_10]
        
        for rank_type, trade in trades_to_plot:
            # Defined Context
            # Buffer: 20 bars before entry, 10 bars after exit
            start_idx = df_sim.index.get_loc(trade['Entry Date'])
            end_idx = df_sim.index.get_loc(trade['Exit Date'])
            
            buffer_before = 30
            buffer_after = 20
            
            plot_start_idx = max(0, start_idx - buffer_before)
            plot_end_idx = min(len(df_sim) - 1, end_idx + buffer_after)
            
            subset = df_sim.iloc[plot_start_idx : plot_end_idx + 1]
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # Subplot 1: Price
            ax1.plot(subset.index, subset['Close'], label='Price', color='black', alpha=0.7)
            
            # Entry/Exit Markers
            ax1.scatter([trade['Entry Date']], [trade['Entry Price']], color='green', marker='^', s=150, label='Entry', zorder=5)
            ax1.scatter([trade['Exit Date']], [trade['Exit Price']], color='red', marker='v', s=150, label='Exit', zorder=5)
            
            # Connect Entry/Exit
            ax1.plot([trade['Entry Date'], trade['Exit Date']], [trade['Entry Price'], trade['Exit Price']], 'k--', alpha=0.3)
            
            title_str = f"{rank_type} Trade | PnL: {trade['PnL %']:.2f}%\nEntry: {trade['Entry Date'].date()} @ {trade['Entry Price']:.2f} | Exit: {trade['Exit Date'].date()} @ {trade['Exit Price']:.2f}"
            ax1.set_title(title_str)
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, linestyle=':')
            
            # Subplot 2: Indicator (Trend Score)
            # Plot Trend Score as bar chart
            scores = subset['trend_score']
            colors = ['lime' if s > 0 else ('red' if s < 0 else 'gray') for s in scores]
            ax2.bar(subset.index, scores, color=colors, width=0.8, label='Trend Score')
            
            # Add Threshold line 0
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
            
            # Mark Entry/Exit Time on Indicator
            ax2.axvline(trade['Entry Date'], color='green', linestyle='--', alpha=0.5)
            ax2.axvline(trade['Exit Date'], color='red', linestyle='--', alpha=0.5)
            
            ax2.set_title('Indicator: Trend Score')
            ax2.set_ylabel('Score')
            ax2.grid(True, linestyle=':')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
    print("PDF generation complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ticker_arg = sys.argv[1]
        file_arg = sys.argv[2]
        run_strategy(ticker_arg, file_arg)
    else:
        run_strategy()
