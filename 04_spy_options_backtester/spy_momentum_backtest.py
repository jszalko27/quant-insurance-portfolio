"""
SPY Momentum Options Backtester
================================
Backtests a rules-based "active sleeve" of a brokerage account under a written risk policy:

  * Weekly contribution split: 60% core (SPY buy & hold) / 40% active sleeve
  * Momentum signal on SPY decides when to hold long calls (or call debit spreads)
  * Risk rules: 5% of sleeve per trade, max 30% of sleeve in open options,
    45-90 DTE entries, defined-risk only, no margin
  * Circuit breakers: sleeve drawdown 25% -> half size, 40% -> pause + review
  * Graduation test: after >= 6 months and >= 20 trades, sleeve must beat
    SPY buy-and-hold after costs with max drawdown < 25%

Option prices are SIMULATED with Black-Scholes, using VIX as implied volatility
(historical SPY option-chain data is expensive). Treat results as a first-pass
filter for ideas, not as a promise of real fills. See LIMITATIONS at the bottom.

Usage
-----
    pip install yfinance pandas numpy scipy matplotlib
    python spy_momentum_backtest.py                       # defaults
    python spy_momentum_backtest.py --signal tsmom --structure spread
    python spy_momentum_backtest.py --structure spread --spread-dollars 7   # small-account version
    python spy_momentum_backtest.py --compare             # all signals x structures
    python spy_momentum_backtest.py --csv spy_vix.csv     # offline: date,spy,vix columns
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm


# ----------------------------------------------------------------------------
# Configuration (mirrors the risk profile; change here or via CLI)
# ----------------------------------------------------------------------------
@dataclass
class Config:
    start: str = "2007-01-01"
    end: str | None = None
    weekly_contribution: float = 500.0   # x sleeve_share 40% = $200 a week into the sleeve
    sleeve_share: float = 0.40           # 40% of each contribution to the active sleeve
    starting_sleeve: float = 0.0         # any lump sum already in the sleeve

    # Signal
    signal: str = "sma200"               # sma200 | sma50 | tsmom | dual
    trade_puts: bool = False             # also buy puts / put spreads on bearish signal

    # Structure
    structure: str = "call"              # call | spread
    target_dte: int = 60                 # enter inside the 45-90 DTE window
    exit_dte: int = 21                   # close before gamma/theta gets ugly
    long_moneyness: float = 1.00         # 1.00 = ATM long strike
    spread_width: float = 0.05           # short strike 5% above long strike (spreads)
    spread_dollars: float | None = None  # fixed $ width instead (e.g. 7) -> much cheaper contracts
    take_profit: float = 1.00            # close at +100% on premium (None to disable)

    # Risk rules
    risk_per_trade: float = 0.05         # 5% of sleeve equity per trade
    max_open_exposure: float = 0.30      # 30% of sleeve in open premium at cost
    dd_half_size: float = 0.25           # sleeve drawdown -> halve size
    dd_pause: float = 0.40               # sleeve drawdown -> pause trading
    review_weeks: int = 4                # weeks paused before resuming (simulated review)
    whole_contracts: bool = True         # False = allow fractional (tests idea, not reality)
    grad_months: int = 6                 # graduation test window
    grad_min_trades: int = 20            # weekly entries cap this at ~26 per 6 months

    # Costs & market
    rate: float = 0.04                   # risk-free rate for pricing & cash yield
    div_yield: float = 0.013
    half_spread_pct: float = 0.03        # 3% of option price lost each side to bid/ask
    commission_per_contract: float = 0.65
    iv_scale: float = 1.00               # multiply VIX (e.g. 0.9 if VIX overstates SPY IV)
    skew: float = -2.5                   # IV(K) = IV_atm * (1 + skew*ln(K/S)); 5% OTM call ~12% lower IV


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(cfg: Config, csv: str | None = None) -> pd.DataFrame:
    if csv:
        df = pd.read_csv(csv, parse_dates=["date"]).set_index("date").sort_index()
        if "spy_tr" not in df:          # no adjusted series: approximate dividends
            yrs = (df.index - df.index[0]).days / 365.25
            df["spy_tr"] = df["spy"] * np.exp(cfg.div_yield * yrs)
        df = df[["spy", "spy_tr", "vix"]]
    else:
        import yfinance as yf
        raw = yf.download(["SPY", "^VIX"], start="2005-01-01", end=cfg.end,
                          auto_adjust=False, progress=False)
        df = pd.DataFrame({"spy": raw["Close"]["SPY"],
                           "spy_tr": raw["Adj Close"]["SPY"],   # dividends reinvested
                           "vix": raw["Close"]["^VIX"]})
    return df.dropna()


def add_signal(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    p = df["spy"]
    sma50, sma200 = p.rolling(50).mean(), p.rolling(200).mean()
    tsmom = p.shift(21) / p.shift(252) - 1          # 12-1 month momentum
    if kind == "sma200":
        sig = np.where(p > sma200, 1, -1)
    elif kind == "sma50":
        sig = np.where(p > sma50, 1, -1)
    elif kind == "tsmom":
        sig = np.where(tsmom > 0, 1, -1)
    elif kind == "dual":                            # both must agree, else flat
        a, b = p > sma200, tsmom > 0
        sig = np.where(a & b, 1, np.where(~a & ~b, -1, 0))
    else:
        raise ValueError(kind)
    df["signal"] = sig
    df.loc[sma200.isna() | tsmom.isna(), "signal"] = 0
    return df


# ----------------------------------------------------------------------------
# Option pricing
# ----------------------------------------------------------------------------
def bs_price(S, K, T, r, q, vol, kind):
    if T <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    vol = max(vol, 1e-4)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


@dataclass
class Position:
    direction: int            # +1 bullish (calls), -1 bearish (puts)
    long_k: float
    short_k: float | None
    expiry: pd.Timestamp
    contracts: float
    entry_price: float        # per-share price paid incl. half spread
    cost: float               # total cash paid incl. commission
    entry_date: pd.Timestamp

    def mid(self, S, today, vol, cfg: Config) -> float:
        T = max((self.expiry - today).days, 0) / 365
        kind = "call" if self.direction > 0 else "put"
        iv = lambda K: max(vol * (1 + cfg.skew * math.log(K / S)), 0.03)
        v = bs_price(S, self.long_k, T, cfg.rate, cfg.div_yield, iv(self.long_k), kind)
        if self.short_k is not None:
            v -= bs_price(S, self.short_k, T, cfg.rate, cfg.div_yield, iv(self.short_k), kind)
        return max(v, 0.0)


# ----------------------------------------------------------------------------
# Backtest engine
# ----------------------------------------------------------------------------
@dataclass
class Result:
    cfg: Config
    daily: pd.DataFrame
    trades: pd.DataFrame
    events: list = field(default_factory=list)


def run(df: pd.DataFrame, cfg: Config) -> Result:
    df = add_signal(df, cfg.signal)
    df = df[df.index >= cfg.start]
    if cfg.end:
        df = df[df.index <= cfg.end]

    cash = cfg.starting_sleeve
    units = cfg.starting_sleeve if cfg.starting_sleeve > 0 else 0.0  # NAV starts at 1.0
    bench_shares = 0.0
    core_shares = 0.0
    contributed = cfg.starting_sleeve
    positions: list[Position] = []
    trades, rows, events = [], [], []
    peak_nav, paused_until = 1.0, None
    daily_cash_yield = cfg.rate / 252

    week_id = df.index.to_period("W")
    last_day_of_week = week_id != np.roll(week_id, -1)
    last_day_of_week[-1] = True

    def sleeve_value(S, today, vol):
        pos_val = sum(p.mid(S, today, vol, cfg) * 100 * p.contracts for p in positions)
        return cash + pos_val

    def close(p: Position, S, today, vol, reason):
        nonlocal cash
        px = p.mid(S, today, vol, cfg) * (1 - cfg.half_spread_pct)
        comm = cfg.commission_per_contract * math.ceil(p.contracts) * (2 if p.short_k else 1)
        proceeds = px * 100 * p.contracts - comm
        cash += proceeds
        trades.append(dict(entry=p.entry_date, exit=today, direction=p.direction,
                           structure="spread" if p.short_k else "single",
                           contracts=p.contracts, cost=p.cost, proceeds=proceeds,
                           pnl=proceeds - p.cost, ret=proceeds / p.cost - 1, reason=reason))

    for i, (today, row) in enumerate(df.iterrows()):
        S, vol, sig = row.spy, row.vix / 100 * cfg.iv_scale, row.signal
        cash *= 1 + daily_cash_yield

        # --- exits -----------------------------------------------------------
        for p in positions[:]:
            dte = (p.expiry - today).days
            val = p.mid(S, today, vol, cfg)
            reason = None
            if dte <= cfg.exit_dte:
                reason = "time"
            elif sig != p.direction:
                reason = "signal_flip"
            elif cfg.take_profit and val >= p.entry_price * (1 + cfg.take_profit):
                reason = "take_profit"
            if reason:
                close(p, S, today, vol, reason)
                positions.remove(p)

        # --- weekly: contributions, breakers, entries -----------------------
        if last_day_of_week[i]:
            nav_units_val = sleeve_value(S, today, vol)
            nav = nav_units_val / units if units > 0 else 1.0
            sleeve_add = cfg.weekly_contribution * cfg.sleeve_share
            core_add = cfg.weekly_contribution - sleeve_add
            cash += sleeve_add
            units += sleeve_add / nav
            contributed += sleeve_add
            core_shares += core_add / row.spy_tr
            bench_shares += sleeve_add / row.spy_tr

            equity = sleeve_value(S, today, vol)
            peak_nav = max(peak_nav, nav)
            dd = 1 - nav / peak_nav

            size_mult = 1.0
            if paused_until is not None and today < paused_until:
                size_mult = 0.0
            elif paused_until is not None and today >= paused_until:
                events.append((today, "resume after review"))
                paused_until, peak_nav = None, nav       # review done: reset the high-water mark
            elif dd >= cfg.dd_pause:
                paused_until = today + pd.Timedelta(weeks=cfg.review_weeks)
                events.append((today, f"PAUSE  (drawdown {dd:.0%})"))
                size_mult = 0.0
            elif dd >= cfg.dd_half_size:
                size_mult = 0.5

            want = sig == 1 or (sig == -1 and cfg.trade_puts)
            open_cost = sum(p.cost for p in positions)
            budget = min(cfg.risk_per_trade * equity * size_mult,
                         cfg.max_open_exposure * equity - open_cost, cash)
            if want and size_mult > 0 and budget > 0:
                direction = int(sig)
                expiry = today + pd.Timedelta(days=cfg.target_dte)
                long_k = round(S * (cfg.long_moneyness if direction > 0 else 2 - cfg.long_moneyness))
                short_k = None
                if cfg.structure == "spread":
                    if cfg.spread_dollars:
                        short_k = long_k + cfg.spread_dollars * direction
                    else:
                        short_k = round(long_k * (1 + cfg.spread_width * direction))
                tmp = Position(direction, long_k, short_k, expiry, 1, 0, 0, today)
                px = tmp.mid(S, today, vol, cfg) * (1 + cfg.half_spread_pct)
                legs = 2 if short_k else 1
                per_contract = px * 100 + cfg.commission_per_contract * legs
                n = budget / per_contract
                n = math.floor(n) if cfg.whole_contracts else n
                if n > 0 and px > 0:
                    cost = n * px * 100 + cfg.commission_per_contract * math.ceil(n) * legs
                    cash -= cost
                    tmp.contracts, tmp.entry_price, tmp.cost = n, px, cost
                    positions.append(tmp)
                elif n == 0:
                    events.append((today, f"SKIP  1 contract ${per_contract:,.0f} > budget ${budget:,.0f}"))

        equity = sleeve_value(S, today, vol)
        rows.append(dict(date=today, spy=S, sleeve=equity, nav=equity / units if units else 1.0,
                         spy_tr=row.spy_tr, bench=bench_shares * row.spy_tr, core=core_shares * row.spy_tr,
                         contributed=contributed, open_positions=len(positions)))

    # close anything left at the end
    last = df.iloc[-1]
    for p in positions:
        close(p, last.spy, df.index[-1], last.vix / 100 * cfg.iv_scale, "end")

    daily = pd.DataFrame(rows).set_index("date")
    return Result(cfg, daily, pd.DataFrame(trades), events)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def max_drawdown(series: pd.Series) -> float:
    return float((1 - series / series.cummax()).max())


def summarize(res: Result) -> dict:
    d, t, cfg = res.daily, res.trades, res.cfg
    years = (d.index[-1] - d.index[0]).days / 365.25
    nav = d["nav"]
    bench_nav = d["spy_tr"] / d["spy_tr"].iloc[0]
    n_trades = len(t)
    wins = t[t.pnl > 0] if n_trades else t
    out = {
        "signal": cfg.signal, "structure": cfg.structure,
        "years": round(years, 1),
        "contributed_to_sleeve": d["contributed"].iloc[-1],
        "sleeve_final": d["sleeve"].iloc[-1],
        "bench_final (same $ into SPY)": d["bench"].iloc[-1],
        "sleeve_CAGR (time-weighted)": nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "SPY_CAGR (w/ dividends)": bench_nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "sleeve_maxDD": max_drawdown(nav),
        "SPY_maxDD": max_drawdown(d["spy_tr"]),
        "trades": n_trades,
        "win_rate": len(wins) / n_trades if n_trades else np.nan,
        "avg_win": wins.ret.mean() if n_trades else np.nan,
        "avg_loss": t[t.pnl <= 0].ret.mean() if n_trades else np.nan,
        "pauses": sum("PAUSE" in e[1] for e in res.events),
        "skipped_entries (too small)": sum("SKIP" in e[1] for e in res.events),
    }
    out["beat_benchmark"] = out["sleeve_final"] > out["bench_final (same $ into SPY)"]
    out["GRADUATES"] = bool(years >= cfg.grad_months / 12 and n_trades >= cfg.grad_min_trades
                            and out["beat_benchmark"]
                            and out["sleeve_maxDD"] < 0.25)
    return out


def graduation_windows(res: Result) -> pd.DataFrame:
    """Rolling check of the 6-month graduation rule, so you see how often it would pass."""
    d, t = res.daily, res.trades
    months, min_trades = res.cfg.grad_months, res.cfg.grad_min_trades
    if t.empty:
        return pd.DataFrame()
    starts = pd.date_range(d.index[0], d.index[-1] - pd.DateOffset(months=months), freq="QS")
    rows = []
    for s in starts:
        e = s + pd.DateOffset(months=months)
        w = d[(d.index >= s) & (d.index < e)]
        if len(w) < 50:
            continue
        sleeve_ret = w.nav.iloc[-1] / w.nav.iloc[0] - 1
        spy_ret = w.spy_tr.iloc[-1] / w.spy_tr.iloc[0] - 1
        n = ((t.entry >= s) & (t.entry < e)).sum()
        dd = max_drawdown(w.nav)
        rows.append(dict(start=s.date(), sleeve=sleeve_ret, spy=spy_ret, trades=n, maxDD=dd,
                         passes=sleeve_ret > spy_ret and dd < 0.25 and n >= min_trades))
    return pd.DataFrame(rows)


def fmt(summary: dict) -> str:
    lines = []
    for k, v in summary.items():
        if isinstance(v, float) and ("CAGR" in k or "DD" in k or "rate" in k or "avg" in k):
            v = f"{v:.1%}"
        elif isinstance(v, float) and k != "years":
            v = f"${v:,.0f}"
        lines.append(f"  {k:<32} {v}")
    return "\n".join(lines)


def plot(res: Result, path: str = "backtest.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = res.daily
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(d.index, d.sleeve, label="Active sleeve")
    ax[0].plot(d.index, d.bench, label="Same $ in SPY")
    ax[0].plot(d.index, d.contributed, label="Contributed", ls="--", c="gray")
    ax[0].set_ylabel("$"); ax[0].legend(); ax[0].set_title(
        f"Sleeve vs SPY  |  signal={res.cfg.signal}, structure={res.cfg.structure}")
    ax[1].fill_between(d.index, -(1 - d.nav / d.nav.cummax()), 0, alpha=.5, label="Sleeve drawdown")
    ax[1].axhline(-res.cfg.dd_half_size, c="orange", ls=":", label="Half-size")
    ax[1].axhline(-res.cfg.dd_pause, c="red", ls=":", label="Pause")
    ax[1].set_ylabel("Drawdown"); ax[1].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="offline data: columns date,spy,vix")
    ap.add_argument("--start", default=Config.start)
    ap.add_argument("--end")
    ap.add_argument("--signal", default=Config.signal, choices=["sma200", "sma50", "tsmom", "dual"])
    ap.add_argument("--structure", default=Config.structure, choices=["call", "spread"])
    ap.add_argument("--spread-dollars", type=float, help="fixed spread width in $ (e.g. 7)")
    ap.add_argument("--puts", action="store_true", help="also trade bearish signals with puts")
    ap.add_argument("--starting-sleeve", type=float, default=0.0)
    ap.add_argument("--fractional", action="store_true",
                    help="allow fractional contracts (tests the idea; not tradable)")
    ap.add_argument("--compare", action="store_true", help="run every signal x structure")
    args = ap.parse_args()

    base = Config(start=args.start, end=args.end, signal=args.signal, structure=args.structure,
                  trade_puts=args.puts, spread_dollars=args.spread_dollars, starting_sleeve=args.starting_sleeve,
                  whole_contracts=not args.fractional)
    df = load_data(base, args.csv)

    if args.compare:
        out = []
        for sig in ["sma200", "sma50", "tsmom", "dual"]:
            for st in ["call", "spread"]:
                cfg = Config(**{**base.__dict__, "signal": sig, "structure": st})
                out.append(summarize(run(df, cfg)))
        table = pd.DataFrame(out)
        pd.set_option("display.width", 200)
        print(table.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        name = f"compare_{args.start[:4]}.csv"
        table.to_csv(name, index=False)
        print(f"\nSaved {name}")
        return

    res = run(df, base)
    print("\n=== SUMMARY ===")
    print(fmt(summarize(res)))
    gw = graduation_windows(res)
    if not gw.empty:
        print(f"\n6-month graduation windows passed: {gw.passes.sum()} / {len(gw)}")
    print("\n=== CIRCUIT BREAKER EVENTS (first 10) ===")
    for e in [e for e in res.events if "SKIP" not in e[1]][:10]:
        print(f"  {e[0].date()}  {e[1]}")
    res.trades.to_csv("trades.csv", index=False)
    print(f"\nSaved trades.csv and {plot(res)}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# LIMITATIONS
# ----------------------------------------------------------------------------
# * Option prices come from Black-Scholes with VIX as ATM IV plus a simple linear
#   skew in log-moneyness; real skew moves with markets.
# * Fills assume a fixed % of price lost to bid/ask; real spreads widen in panics.
# * Weekly decisions use that day's close; no intraday stops.
# * Before trusting a result, sanity-check it on real option chain data
#   (e.g., a broker's paper-trading account) during the 4-8 week paper period.
