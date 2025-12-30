import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def download_data():
    """
    Downloads BTC-USD data from yfinance from 2018-01-01 to today.
    Saves as CSV to data/btcusd.csv.
    """
    ticker = "BTC-USD"
    start_date = "2015-01-01"
    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading {ticker} from {start_date} to {today}...")
    
    try:
        data = yf.download(ticker, start=start_date, end=today)
        
        if data.empty:
            print("No data data found.")
            return

        # Handle potentially multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Standardize columns
        # Expected: Date, Open, High, Low, Close, Volume
        # Date is usually index
        
        # Ensure we have the right columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            print(f"Missing columns. Available: {data.columns}")
            # Try to handle 'Adj Close' if 'Close' is missing or map it
            # But yfinance usually provides 'Close'
        
        # Save to CSV
        output_file = os.path.join("data", "btcusd.csv")
        data.to_csv(output_file)
        print(f"Successfully saved data to {output_file}")
        print(f"Rows: {len(data)}")
        print(data.tail())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_data()
