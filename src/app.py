from flask import Flask, render_template, jsonify
from web_utils import get_eth_data_with_live, calculate_indicators
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    print("Received request for /api/data")
    # Load and process data
    df = get_eth_data_with_live()
    print(f"Data Loaded: {len(df)} rows")
    
    if df.empty:
        return jsonify({"error": "No data available"}), 500
        
    # Calculate indicators
    df = calculate_indicators(df)
    print("Indicators calculated")
    
    # Filter for display (e.g., last 6 months to match recent plot)
    # Or send more and let frontend filter? 
    # Let's send last 365 days for flexibility, frontend can zoom.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    mask = df.index >= start_date
    df_subset = df[mask].copy()
    
    # Prepare JSON structure
    # Arrays for Plotly
    dates = df_subset.index.strftime('%Y-%m-%d').tolist()
    
    # Calculate signal changes for markers
    # signal_change = 1 (Entry), -1 (Exit)
    # We already have 'signal' column from calculate_indicators?
    # Let's verify calculate_indicators returns it. 
    # It does: df['signal_change'] = df['signal'].diff()
    
    # Extract entries and exits
    # We need arrays matching the *filtered* subset
    df_subset['signal_change'] = df_subset['signal'].diff()
    
    # Entry Points: signal_change == 1
    entries = df_subset[df_subset['signal_change'] == 1]
    entry_dates = entries.index.strftime('%Y-%m-%d').tolist()
    entry_prices = entries['Low'].tolist() # Buy at Low? Or Close? Visualization usually Low
    
    # Exit Points: signal_change == -1
    exits = df_subset[df_subset['signal_change'] == -1]
    exit_dates = exits.index.strftime('%Y-%m-%d').tolist()
    exit_prices = exits['High'].tolist() # Sell at High
    
    data = {
        "dates": dates,
        "open": df_subset['Open'].tolist(),
        "high": df_subset['High'].tolist(),
        "low": df_subset['Low'].tolist(),
        "close": df_subset['Close'].tolist(),
        "trend_score": df_subset['trend_score'].tolist(),
        "signal": df_subset['signal'].tolist(),
        "last_price": df_subset['Close'].iloc[-1],
        "last_date": dates[-1],
        "current_score": int(df_subset['trend_score'].iloc[-1]),
        "current_signal": "LONG" if df_subset['trend_score'].iloc[-1] > 0 else "FLAT",
        "events": {
            "entries": {"dates": entry_dates, "prices": entry_prices},
            "exits": {"dates": exit_dates, "prices": exit_prices}
        }
    }
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=False, port=5001)
