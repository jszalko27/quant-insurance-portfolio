<div align="center">

# 📉 Systematic Options Strategy Backtester

**Can a rules-based momentum strategy that buys SPY call options beat simply holding SPY under strict risk limits?**

![Python](https://img.shields.io/badge/Python-3.11+-2a78d6?logo=python&logoColor=white)
![Pricing](https://img.shields.io/badge/pricing-Black--Scholes%20+%20skew-1baf7a)
![Data](https://img.shields.io/badge/data-19%20years%20SPY%20%2B%20VIX-eda100)

</div>

---

## 📌 At a glance

| | |
|:--|:--|
| **Variants tested** | 4 signals × 2 structures × 2 periods |
| **Best variant** | **2.1%/yr** (vs **13.9%** for SPY since 2010) |
| **Worst drawdowns** | 57–80% (vs 34% for SPY since 2010) |
| **Key lesson** | Found a pricing bias that faked a 2–3 point edge |

> [!IMPORTANT]
> **No variant beat buy-and-hold.** The first version *appeared* to show call spreads winning by 2–3 points a year. That came from pricing the sold call at the same volatility as the bought one. Adding realistic volatility skew erased the edge. Checking a too-good result is the real skill this project shows.

---

## 🧭 How it works

```mermaid
flowchart LR
    A[SPY + VIX<br/>daily, 2007–2026] --> B[Momentum signal<br/>200-day · 50-day · 12-1 · combined]
    B --> C{Bullish?}
    C -- yes --> D[Buy 60-day call<br/>or call spread]
    C -- no --> E[Stay in cash]
    D --> F[Risk rules<br/>5% per trade · 30% max open<br/>drawdown breakers]
    F --> G[Exit at 21 days<br/>or signal flip]
    G --> H[Compare with SPY<br/>dividends reinvested]
```

---

## 📈 Results

![Options vs SPY](figures/options_vs_spy.png)

| Signal | 2007–2026 | 2010–2026 |
|:--|--:|--:|
| 200-day moving average | −1.2% | −1.2% |
| 50-day moving average | 0.2% | 2.1% |
| 12-1 month momentum | −5.5% | −3.5% |
| Combined | −0.3% | 1.7% |
| **SPY buy & hold** | **10.8%** | **13.9%** |

*Annual return of the better structure (calls or spreads) for each signal.*

> [!TIP]
> **Lessons**
> 1. Buying options repeatedly pays time decay and the implied-volatility premium every cycle; momentum timing doesn't earn it back.
> 2. Model both legs of a spread carefully: flat volatility across strikes overstated spread returns by 2–3 points a year.
> 3. Contract size interacts with position sizing: one at-the-money SPY call costs about $1,700–2,300, so a 5% rule needs a $33k+ sleeve.

**Follow-up:** the same signal applied to shares instead of options works as risk control. See [project 05](../05_trend_following/).

---

## 🔬 Method details

- **Pricing:** Black-Scholes with VIX as at-the-money implied volatility plus a log-moneyness skew, $\sigma(K) = \sigma_{ATM}\,(1 - 2.5\ln(K/S))$
- **Costs:** 3% of option price lost to bid/ask each way, plus $0.65 a contract
- **Risk rules:** 5% of the sleeve per trade, 30% maximum open premium, size halved at a 25% drawdown, trading paused at 40%
- **Benchmark:** the same contributions into SPY with dividends reinvested

---

## 📁 Files

```
04_spy_options_backtester/
├── spy_momentum_backtest.py   # backtester (downloads SPY and VIX via yfinance)
├── figures/                   # README chart
└── output/                    # compare_2007.csv, compare_2010.csv
```

```bash
pip install yfinance pandas numpy scipy matplotlib
python3 spy_momentum_backtest.py --compare
python3 spy_momentum_backtest.py --compare --start 2010-01-01
```
