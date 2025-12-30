# BTC Data Project

## Video reference
https://www.youtube.com/watch?v=2LC979UiRns

## Description
This project automates the downloading of BTC-USD historical data.

## Project Structure
- `src/`: Source code
- `data/`: Downloaded CSV data
- `plots/`: Plots (empty)
- `results/`: Analysis results (empty)
- `logs/`: Logs (empty)
- `.venv/`: Python virtual environment

## Setup
1. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Install dependencies (if not already):
   ```bash
   pip install yfinance
   ```

## Usage
Run the download script:
```bash
python src/download_btc.py
```
This will download BTC-USD data from 2015-01-01 to today and save it to `data/btcusd.csv`.
