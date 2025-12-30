# ETH Strategy Improvement Roadmap

Given the "Mystic Pulse" strategy's success in capturing ETH's high-beta trends while limiting drawdowns, the next phase of optimization should focus on **Dynamic Risk Management** and **Regime Filtering**.

## 1. Dynamic Volatility Targeting (Risk-Parity)
**Why**: ETH volatility explodes during manias (2017, 2021) and crashes. A fixed 100% position size is suboptimal.
- **Concept**: Target a constant annualized volatility (e.g., 50%).
- **Implementation**: `Position Size = Target Vol / Current Vol`.
    - When volatility is LOW (quiet accumulation), size > 100% (safe leverage).
    - When volatility is EXTREME (mania peak), size < 100% (automatically taking profits).
- **Expected Outcome**: Higher risk-adjusted returns (Sharpe Ratio > 1.5).

## 2. Regime Filter: ETH/BTC Ratio
**Why**: ETH typically outperforms only when it is gaining strength against Bitcoin.
- **Concept**: Only take LONG signals if `ETH/BTC` is above its 50-day SMA.
- **Logic**: If ETH is weak against BTC, even if USD trend is up, it might be a "drag along" rally. Ensure ETH is the *leader*.
- **Expected Outcome**: filtering out weak, choppy uptrends that often fail.

## 3. Parabolic Trailing Stop (Chandelier Exit)
**Why**: The current ADX/DI exit is somewhat "laggy"—it waits for the trend to actually reverse. In parabolic runs, this gives back too much open profit (e.g., 20-30% drops from peak).
- **Concept**: Activate a tighter trailing stop (e.g., 3x ATR from High) *only* when Trend Score > 2 (indicating extreme momentum).
- **Expected Outcome**: Locking in more profit near the absolute top of mania spikes.

## 4. Modest Leverage (1.2x - 1.5x)
**Why**: You have successfully reduced Max Drawdown from -94% (B&H) to -52%.
- **Concept**: Since the tail risk is cut, you have "room" to add leverage while keeping total risk within acceptable limits (e.g., -60% Max DD).
- **Implementation**: Use a 1.25x or 1.5x leverage factor on the base signal.
- **Expected Outcome**: Potentially boosting Total Return from ~8,400% to **20,000%+** without exceeding the risk profile of holding raw BTC.

## 5. Short Strategy (Hedging)
**Why**: The current strategy is "Long/Flat". It sits in cash during crashes.
- **Concept**: Implement a "Light Short" (-0.5x) when Trend Score is deeply negative (<-2).
- **Expected Outcome**: Profiting from the massive -94% crash phases rather than just avoiding them.

---
**Recommended First Step**: Implement **#2 (ETH/BTC Filter)** as it is a logic change, not a risk-management change, and offers the highest "purity" upgrade.
