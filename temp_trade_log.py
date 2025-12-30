import pandas as pd
df = pd.read_csv('results/mystic_pulse_ETH-USD_trades.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_2025 = df[df['Date'].dt.year == 2025].copy()
trades = []
current_trade = {}
for _, row in df_2025.iterrows():
    if row['Type'] == 'Buy':
        current_trade = {'Entry': row['Date'], 'Entry Price': row['Price']}
    elif row['Type'] == 'Sell' and current_trade:
        current_trade['Exit'] = row['Date']
        current_trade['Exit Price'] = row['Price']
        current_trade['Equity'] = row['Equity']
        current_trade['PnL'] = (row['Price'] - current_trade['Entry Price']) / current_trade['Entry Price'] * 100
        trades.append(current_trade)
        current_trade = {}

print(f'{len(trades)} trades found in 2025\n')
print(f'| Entry Date | Entry Price | Exit Date | Exit Price | PnL % | Equity |')
print(f'|---|---|---|---|---|---|')
for t in trades:
    print(f'| {t["Entry"].strftime("%Y-%m-%d")} | ${t["Entry Price"]:.2f} | {t["Exit"].strftime("%Y-%m-%d")} | ${t["Exit Price"]:.2f} | {t["PnL"]:.2f}% | ${t["Equity"]:.2f} |')
