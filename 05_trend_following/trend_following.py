"""
Trend following with shares: hold SPY when it is above its moving average, T-bills when below.

Tests whether the momentum signal that failed with options (project 04) works when you hold
the underlying instead, which avoids paying option time decay. Variants:

  buy_hold      SPY total return, always invested (benchmark)
  sma200        SPY above its 200-day average -> SPY, else T-bills
  sma10m        month-end check: SPY above its 10-month average -> SPY, else T-bills
  dual          both the 200-day average and 12-1 month momentum must agree to be invested
  sma200_2x     sma200, but 2x daily SPY exposure when invested (simulated leveraged ETF:
                2 x SPY return - borrowing at the T-bill rate - 0.9%/yr fee)

Rules that keep the test honest:
  * The signal uses today's close; the position changes at TOMORROW's close (no look-ahead).
  * Each switch costs 0.05% (spread + slippage).
  * SPY uses dividend-adjusted prices; T-bills earn the 13-week rate (^IRX).
  * Taxes are ignored (switching creates short-term gains in a taxable account).

    pip install yfinance pandas numpy matplotlib
    python trend_following.py                  # 1994 to today
    python trend_following.py --start 2007-01-01
    python trend_following.py --csv data.csv   # offline: columns date, spy_tr, irx (annual %, e.g. 4.3)
"""
from __future__ import annotations

import argparse, json, pathlib
import numpy as np
import pandas as pd

COST = 0.0005          # per switch
LEV_FEE = 0.009        # annual expense of a 2x fund
OUT = pathlib.Path("output")


def load(start, end=None, csv=None):
    if csv:
        df = pd.read_csv(csv, parse_dates=["date"]).set_index("date").sort_index()
    else:
        import yfinance as yf
        raw = yf.download(["SPY", "^IRX"], start="1993-01-01", end=end, auto_adjust=False, progress=False)
        df = pd.DataFrame({"spy_tr": raw["Adj Close"]["SPY"], "irx": raw["Close"]["^IRX"]})
    df["irx"] = df["irx"].ffill()
    df = df.dropna()
    df["r_spy"] = df.spy_tr.pct_change()
    df["r_bill"] = df.irx / 100 / 252
    return df[df.index >= start].dropna()


def signals(full: pd.DataFrame) -> pd.DataFrame:
    p = full.spy_tr
    s = pd.DataFrame(index=full.index)
    s["buy_hold"] = 1.0
    s["sma200"] = (p > p.rolling(200).mean()).astype(float)
    monthly = p.resample("ME").last()
    m_sig = (monthly > monthly.rolling(10).mean()).astype(float)
    s["sma10m"] = m_sig.reindex(full.index, method="ffill")
    mom = p.shift(21) / p.shift(252) - 1
    s["dual"] = ((p > p.rolling(200).mean()) & (mom > 0)).astype(float)
    s["sma200_2x"] = s["sma200"] * 2
    return s.fillna(0)


def backtest(df: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    pos = sig.shift(1).fillna(0)                  # act on yesterday's signal: no look-ahead
    rets = {}
    for name in sig.columns:
        w = pos[name]
        invested = np.minimum(w, 1.0)
        lev_extra = np.maximum(w - 1.0, 0.0)       # extra exposure is borrowed at the bill rate
        r = invested * df.r_spy + (1 - invested) * df.r_bill
        r = r + lev_extra * (df.r_spy - df.r_bill) - (w > 1) * LEV_FEE / 252
        switches = w.diff().abs().fillna(0) > 0
        r = r - switches * COST
        rets[name] = r
    return pd.DataFrame(rets)


def stats(r: pd.Series, rf: pd.Series, pos: pd.Series) -> dict:
    nav = (1 + r).cumprod()
    yrs = len(r) / 252
    ex = r - rf
    return {
        "CAGR": nav.iloc[-1] ** (1 / yrs) - 1,
        "volatility": r.std() * np.sqrt(252),
        "sharpe": ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else np.nan,
        "max_drawdown": float((1 - nav / nav.cummax()).max()),
        "worst_year": float(r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1).min()),
        "time_invested": float((pos > 0).mean()),
        "switches_per_year": float((pos.diff().abs() > 0).sum() / yrs),
        "growth_of_$10k": 10000 * nav.iloc[-1],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="1994-01-01")
    ap.add_argument("--end")
    ap.add_argument("--csv")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    full = load("1900-01-01", a.end, a.csv)       # full history so averages are warm at --start
    sig = signals(full)
    df = full[full.index >= a.start]
    sig = sig.loc[df.index]
    rets = backtest(df, sig)
    pos = sig.shift(1).fillna(0)
    table = pd.DataFrame({k: stats(rets[k], df.r_bill, pos[k]) for k in rets}).T
    pd.set_option("display.width", 160)
    fmt = table.copy()
    for c in ["CAGR", "volatility", "max_drawdown", "worst_year", "time_invested"]:
        fmt[c] = (fmt[c] * 100).map("{:.1f}%".format)
    fmt["sharpe"] = fmt["sharpe"].map("{:.2f}".format)
    fmt["switches_per_year"] = fmt["switches_per_year"].map("{:.1f}".format)
    fmt["growth_of_$10k"] = fmt["growth_of_$10k"].map("${:,.0f}".format)
    print(f"\n{df.index[0].date()} to {df.index[-1].date()}\n")
    print(fmt.to_string())
    table.to_csv(OUT / f"trend_stats_{a.start[:4]}.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for k, col in [("buy_hold", "#888888"), ("sma200", "#2a6f97"), ("dual", "#81b29a"), ("sma200_2x", "#e07a5f")]:
        nav = (1 + rets[k]).cumprod()
        axs[0].plot(nav.index, nav, label=k, color=col, lw=1.4)
        axs[1].plot(nav.index, -(1 - nav / nav.cummax()) * 100, color=col, lw=1)
    axs[0].set_yscale("log"); axs[0].set_ylabel("Growth of $1 (log)"); axs[0].legend(frameon=False)
    axs[0].set_title("Trend following vs buy and hold")
    axs[1].set_ylabel("Drawdown (%)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / f"trend_following_{a.start[:4]}.png", dpi=150)
    print(f"\nSaved output/trend_stats_{a.start[:4]}.csv and output/trend_following_{a.start[:4]}.png")


if __name__ == "__main__":
    main()
