"""
Regenerate every README figure in one consistent visual style.

    python3 make_figures.py          # writes <project>/figures/*.png

Requires each project's outputs to exist (run the projects first; see README.md).
"""
import json, pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = pathlib.Path(__file__).parent
# Colorblind-validated categorical palette (fixed order, never cycled)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SEQ = ["#e8f1fb", "#c3dbf5", "#8fbbec", "#5a98e0", "#2a78d6", "#1d5aa6", "#123d73"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "lines.linewidth": 2, "text.parse_math": False,
})
money = FuncFormatter(lambda v, _: f"${v:,.0f}M")
pct = FuncFormatter(lambda v, _: f"{v:.0f}%")


def save(fig, project, name, subtitle=None):
    out = ROOT / project / "figures"; out.mkdir(exist_ok=True)
    if subtitle:
        fig.text(0.012, 0.005, subtitle, fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if subtitle else 0, 1, 1))
    fig.savefig(out / name, dpi=160); plt.close(fig)
    print("wrote", project, name)


def bar_label(ax, bars, fmt, inside=False, color=INK):
    for b in bars:
        w = b.get_width() if b.get_width() != 0 else b.get_height()
        if b.get_width() > b.get_height():   # horizontal
            ax.text(b.get_x() + b.get_width() + 0.4, b.get_y() + b.get_height() / 2, fmt(b.get_width()),
                    va="center", fontsize=9, color=color)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, fmt(b.get_height()),
                    ha="center", va="bottom", fontsize=9, color=color)


# ------------------------------------------------------------------------- 01
def accumulation():
    P = "01_cyber_accumulation_model"
    sys.path.insert(0, str(ROOT / P))
    import accumulation_model as am
    a = am.Assumptions(n_years=50000)
    pf, by_source, max_occ, ev, occ = am.simulate(a)
    s = am.summarize(a, pf, by_source, max_occ, ev)
    ri = am.reinsurance_and_capital(a, by_source, occ, s["premium_$M"] * 1e6)
    series = ri.pop("_series")
    stress = am.stress_scenarios(a, pf)
    total = by_source.sum(axis=1).values / 1e6

    # EP curves
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for vals, lab, col, ls in [(total, "Gross: all events in a year (AEP)", BLUE, "-"),
                               (series["net"] / 1e6, "Net of reinsurance (AEP)", AQUA, "-"),
                               (max_occ / 1e6, "Largest single event (OEP)", ORANGE, "--")]:
        srt = np.sort(vals)[::-1]
        rp = a.n_years / np.arange(1, len(srt) + 1)
        m = rp >= 1
        ax.plot(rp[m], srt[m], color=col, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_xlim(1, 2000)
    ax.set_xticks([1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]); ax.set_xticklabels(["1", "2", "5", "10", "25", "50", "100", "200", "500", "1,000"])
    ax.yaxis.set_major_formatter(money)
    g200 = am.return_period_loss(total, 200)
    ax.axvline(200, color=INK2, lw=0.8, ls=":")
    ax.annotate(f"1-in-200 gross: ${g200:.0f}M", (200, g200), xytext=(250, g200 - 40), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("Return period (years)"); ax.set_ylabel("Annual insured loss")
    ax.set_title("How bad can a year get? Loss by return period")
    ax.legend(loc="upper left")
    save(fig, P, "ep_curves.png", "5,000 synthetic policies, 50,000 simulated years. Reinsurance: $40M xs $20M per systemic event + 20% quota share.")

    # Average vs tail mix
    att = ["BEC", "Funds transfer fraud", "Ransomware", "Data breach", "Other"]
    groups = {"Single-company incidents": att, "Software supply chain": ["Software supply chain"],
              "Cloud outage": ["Cloud outage"], "Ransomware campaign": ["Ransomware campaign"]}
    tail_years = total >= np.quantile(total, 1 - 1 / 200)
    avg = {g: by_source[c].sum(axis=1).mean() for g, c in groups.items()}
    tl = {g: by_source.loc[tail_years, c].sum(axis=1).mean() for g, c in groups.items()}
    A = pd.Series(avg) / sum(avg.values()) * 100; T = pd.Series(tl) / sum(tl.values()) * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(A))[::-1]; h = 0.36
    b1 = ax.barh(y + h / 2, A.values, h, color=BLUE, label="Average year")
    b2 = ax.barh(y - h / 2, T.values, h, color=ORANGE, label="Worst 1-in-200 years")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.0f}%", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(A.index); ax.xaxis.set_major_formatter(pct)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlim(0, 95); ax.set_title("Single-company attacks drive the average; shared vendors drive the tail")
    ax.legend(loc="lower right")
    save(fig, P, "average_vs_tail.png", "Share of insured loss by source.")

    # Vendor concentration
    vt = pd.Series(s["vendor_tail_contribution_$M"]).rename({"Campaign": "Ransomware campaign"}).sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    cols = [ORANGE if k.startswith("Software") else BLUE if k.startswith("Cloud") else YELLOW for k in vt.index]
    bars = ax.barh(vt.index, vt.values, color=cols, height=0.6)
    for b in bars:
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2, f"${b.get_width():.1f}M", va="center", fontsize=9)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True); ax.xaxis.set_major_formatter(money)
    ax.set_title("One software product used by 20% of insureds is the biggest tail driver")
    save(fig, P, "vendor_concentration.png", "Average loss from each shared dependency in 1-in-200+ years. Orange = software, blue = cloud, yellow = ransomware campaigns.")

    # Stress scenarios
    st = pd.Series(stress)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = [k.replace(", ", "\n") for k in st.index]
    cols = [BLUE if "Cloud" in k else ORANGE for k in st.index]
    bars = ax.bar(range(len(st)), st.values, color=cols, width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"${b.get_height():.0f}M", ha="center", fontsize=9)
    ax.set_xticks(range(len(st))); ax.set_xticklabels(labels, fontsize=8.5); ax.yaxis.set_major_formatter(money)
    ax.set_title("Stress tests: a 24-hour cloud outage costs $1M, a 72-hour one $71M")
    save(fig, P, "stress_tests.png", "Deterministic scenarios. The 12-hour waiting period absorbs short outages.")

    # Capital
    g, n = ri["gross"], ri["net"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    vals = [g["required_capital_$M"], n["required_capital_$M"]]
    bars = ax.bar(["Gross (no reinsurance)", "Net of reinsurance"], vals, color=[BLUE, AQUA], width=0.5)
    for b, roc in zip(bars, [g["return_on_capital"], n["return_on_capital"]]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"${b.get_height():.0f}M\nreturn on capital {roc:.1%}",
                ha="center", fontsize=9)
    ax.yaxis.set_major_formatter(money); ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("Reinsurance cuts required capital by more than half")
    save(fig, P, "capital.png", "Capital = 1-in-200 underwriting loss (99.5% VaR).")
    sys.path.pop(0)


# ------------------------------------------------------------------------- 02
def pricing():
    P = "02_cyber_pricing_glm"; O = ROOT / P / "output"
    c = pd.read_csv(O / "sql_loss_ratio_by_controls.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(c.controls_in_place.astype(str), c.loss_ratio * 100, color=BLUE, width=0.6)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", fontsize=9)
    ax.axhline(62, color=INK2, ls="--", lw=1); ax.text(4.45, 64, "62% target", fontsize=8.5, color=INK2, ha="right")
    ax.yaxis.set_major_formatter(pct); ax.set_xlabel("Security controls in place (MFA, EDR, backups, patching)")
    ax.set_title("The current plan overcharges well-protected firms")
    save(fig, P, "loss_ratio_by_controls.png", "Loss ratio = claims / premium, from SQL on 180,000 synthetic policies (2023-2025).")

    for kind, fname, title in [("frequency", "freq", "Frequency"), ("severity", "sev", "Severity")]:
        d = pd.read_csv(O / f"{kind}_fitted_vs_true.csv")
        d = d[d.term != "log revenue / $25M"].iloc[::-1].reset_index(drop=True)
        d["term"] = d.term.str.replace("industry: ", "").str.replace("_", " ")
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 1.2))
        y = np.arange(len(d))
        ax.hlines(y, d.low_95, d.high_95, color=BLUE, lw=2)
        ax.plot(d.fitted, y, "o", color=BLUE, ms=7, label="Fitted, with 95% interval")
        ax.plot(d.true, y, "D", color=ORANGE, ms=6, mfc=SURF, mew=2, label="True value")
        ax.axvline(1, color=INK2, lw=0.8, ls=":")
        ax.set_yticks(y); ax.set_yticklabels(d.term); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_xlabel("Relativity (1.0 = no effect)")
        ax.set_title(f"{title} model recovers the true risk factors")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)
        save(fig, P, f"relativities_{fname}.png", "Industry relative to professional services.")

    lift = pd.read_csv(O / "lift_chart.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lift.decile, lift.pred_loss_cost, "o-", color=BLUE, label="Predicted")
    ax.plot(lift.decile, lift.actual_loss_cost, "s--", color=ORANGE, label="Actual")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xticks(range(1, 11)); ax.set_xlabel("Decile of predicted risk (1 = safest)"); ax.set_ylabel("Loss per policy-year")
    ax.set_title("Lift chart on 2025 holdout: riskiest 10% cost about 80x the safest")
    ax.legend(loc="upper left")
    save(fig, P, "lift_chart.png", "Trained on 2023-2024, tested on 2025 (out of time).")

    dl = pd.read_csv(O / "double_lift.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(dl) + 1); w = 0.36
    b1 = ax.bar(x - w / 2, dl.current_plan_LR * 100, w, color=INK2, alpha=0.45, label="Current plan")
    b2 = ax.bar(x + w / 2, dl.glm_plan_LR * 100, w, color=BLUE, label="GLM plan")
    ax.axhline(62, color=INK2, ls="--", lw=1)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["GLM says\nmuch cheaper", "2", "3", "4", "GLM says\nmuch dearer"])
    ax.yaxis.set_major_formatter(pct); ax.set_ylim(0, 115); ax.legend(loc="upper left")
    ax.set_title("Double lift: current plan swings from 27% to 94% loss ratio")
    save(fig, P, "double_lift.png", "Policies sorted by GLM premium / current premium. A flat profile near 62% = accurate pricing.")


# ------------------------------------------------------------------------- 03
def reserving():
    P = "03_cyber_loss_reserving"; O = ROOT / P / "output"
    sys.path.insert(0, str(ROOT / P))
    import reserving as rv
    paid = pd.read_csv(O / "paid_triangle.csv", index_col=0)
    ult = pd.read_csv(O / "ultimates_by_method.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pct_paid = paid.values / ult.true_ultimate.values[:, None] * 100
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(np.ma.masked_invalid(pct_paid), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(pct_paid.shape[0]):
        for j in range(pct_paid.shape[1]):
            v = pct_paid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{paid.values[i, j]/1e6:.1f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 60 else INK)
    ax.set_xticks(range(paid.shape[1])); ax.set_xticklabels(paid.columns); ax.set_yticks(range(len(paid))); ax.set_yticklabels(paid.index)
    ax.set_xlabel("Development year"); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("% of ultimate paid", color=INK2); cb.outline.set_visible(False)
    ax.set_title("Paid loss triangle ($M): the blank corner is what we must estimate")
    save(fig, P, "paid_triangle.png", "Each row is an accident year; each column a year of development. Shading = share of the true ultimate paid so far.")

    methods = {"CL_paid": ("Chain ladder, paid", ORANGE), "CL_incurred": ("Chain ladder, incurred", BLUE),
               "BF_paid": ("Bornhuetter-Ferguson, paid", YELLOW), "BF_incurred": ("Bornhuetter-Ferguson, incurred", AQUA)}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for m, (lab, col) in methods.items():
        ax.plot(ult.index, ult[f"{m}_err%"], "o-", color=col, label=lab, ms=5)
    ax.axhline(0, color=INK, lw=0.8); ax.yaxis.set_major_formatter(pct)
    ax.set_xticks(ult.index); ax.set_ylabel("Error vs true ultimate")
    ax.set_title("All methods agree on old years; the newest year is where they split")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    save(fig, P, "method_errors.png")

    true_ult, paid_df, _ = rv.simulate()
    boot, _ = rv.odp_bootstrap(paid_df.values)
    cur, _ = rv.latest(paid_df.values)
    true_res = (true_ult - cur).sum() / 1e6
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(boot / 1e6, bins=60, color=BLUE, alpha=0.85, edgecolor=SURF, linewidth=0.5)
    top = ax.get_ylim()[1]
    for v, lab, col, ls, yf, ha in [(np.median(boot) / 1e6, "Median", INK, "--", 0.97, "right"), (true_res, "True reserve", ORANGE, "-", 0.97, "left"),
                                    (np.percentile(boot, 99.5) / 1e6, "99.5th percentile", INK2, ":", 0.6, "left")]:
        ax.axvline(v, color=col, ls=ls, lw=2 if col == ORANGE else 1.2)
        ax.text(v + (0.5 if ha == "left" else -0.5), top * yf, f"{lab}\n${v:.0f}M", fontsize=8.5, color=INK, va="top", ha=ha,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none"))
    ax.xaxis.set_major_formatter(money); ax.set_ylabel("Simulations")
    ax.set_title("Bootstrap: the reserve is a range, not a number")
    save(fig, P, "bootstrap.png", "5,000 over-dispersed Poisson bootstrap simulations of the paid chain ladder.")
    sys.path.pop(0)


# ------------------------------------------------------------------------- 04
def options():
    P = "04_spy_options_backtester"; O = ROOT / P / "output"
    names = {"sma200": "200-day avg", "sma50": "50-day avg", "tsmom": "12-1 momentum", "dual": "Combined"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (f, period) in zip(axs, [("compare_2007.csv", "2007-2026"), ("compare_2010.csv", "2010-2026")]):
        d = pd.read_csv(O / f)
        spy = d["SPY_CAGR (w/ dividends)"].iloc[0] * 100
        x = np.arange(4); w = 0.36
        calls = d[d.structure == "call"].set_index("signal").loc[list(names)]["sleeve_CAGR (time-weighted)"] * 100
        spr = d[d.structure == "spread"].set_index("signal").loc[list(names)]["sleeve_CAGR (time-weighted)"] * 100
        ax.bar(x - w / 2, calls, w, color=BLUE, label="Long calls")
        ax.bar(x + w / 2, spr, w, color=AQUA, label="Call spreads")
        ax.axhline(spy, color=ORANGE, lw=2, label="SPY buy & hold")
        ax.text(3.45, spy + 0.6, f"SPY {spy:.1f}%", ha="right", fontsize=9, color=INK)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(names.values(), fontsize=8.5)
        ax.set_title(period, fontsize=11); ax.yaxis.set_major_formatter(pct)
    for ax in axs: ax.set_ylim(-10, 17)
    axs[0].set_ylabel("Annual return"); axs[0].legend(loc="center left", bbox_to_anchor=(0.0, 0.6), fontsize=8)
    fig.suptitle("Every options variant lost to simply holding SPY", x=0.01, ha="left", fontweight="bold", fontsize=12)
    save(fig, P, "options_vs_spy.png", "Time-weighted annual return of the options sleeve vs SPY with dividends. Realistic volatility skew applied.")


# ------------------------------------------------------------------------- 05
def trend():
    P = "05_trend_following"; O = ROOT / P / "output"
    labels = {"buy_hold": "Buy & hold", "sma200": "200-day avg", "sma10m": "10-month avg",
              "dual": "Combined", "sma200_2x": "200-day avg, 2x"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (f, period) in zip(axs, [("trend_stats_1994.csv", "1994-2026"), ("trend_stats_2007.csv", "2007-2026")]):
        d = pd.read_csv(O / f, index_col=0)
        for k, lab in labels.items():
            col = INK2 if k == "buy_hold" else ORANGE if k.endswith("2x") else BLUE
            ax.scatter(d.loc[k, "max_drawdown"] * 100, d.loc[k, "CAGR"] * 100, s=90, color=col, zorder=3,
                       edgecolor=SURF, linewidth=1.5)
            ax.annotate(lab, (d.loc[k, "max_drawdown"] * 100, d.loc[k, "CAGR"] * 100), xytext=(6, 4),
                        textcoords="offset points", fontsize=8.5)
        ax.set_xlabel("Worst drawdown (lower is better)"); ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
        ax.set_title(period, fontsize=11); ax.grid(axis="x", visible=True); ax.set_xlim(15, 62)
    axs[0].set_ylabel("Annual return")
    fig.suptitle("Trend following halves the worst drawdown for 1-3 points of return", x=0.01, ha="left", fontweight="bold", fontsize=12)
    save(fig, P, "risk_return.png", "SPY with dividends vs 13-week T-bills. Next-day execution, 0.05% per switch, taxes ignored.")


if __name__ == "__main__":
    for f in (accumulation, pricing, reserving, options, trend):
        f()
