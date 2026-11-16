"""
Cyber Catastrophe Accumulation Model  (v2)
==========================================
v2 adds: annual aggregate policy limits, per-event cat excess-of-loss and quota-share
reinsurance, gross vs net views, and capital / return-on-capital metrics.

A teaching / portfolio model of how a cyber insurance book can lose money all at once.

It simulates a synthetic portfolio of commercial cyber policies for many years and
separates two kinds of loss:

  1. ATTRITIONAL   - independent, single-company incidents (BEC, funds-transfer fraud,
                     ransomware, data breach). These drive day-to-day loss ratios.
  2. SYSTEMIC      - one event that hits many policyholders at once because they share a
                     technology dependency:
                       * Cloud provider outage    -> business interruption (BI) for every
                                                     dependent insured past the waiting period
                       * Software supply-chain    -> a compromised vendor product (file
                         compromise                  transfer, remote management, VPN, ...)
                                                     leads to breaches at a share of its users
                       * Mass ransomware campaign -> a widespread exploited vulnerability hits
                                                     unpatched / poorly defended insureds

Outputs: average annual loss (AAL) by peril, aggregate & occurrence exceedance-probability
curves (AEP / OEP), 1-in-N return-period losses (PML), TVaR, tail attribution by source and
by vendor, deterministic stress scenarios, and a security-control sensitivity test.

All vendors are generic ("Cloud A", "Software S1") and all parameters are illustrative
assumptions loosely calibrated to public claims studies (see methodology doc). This is a
learning model, not a pricing tool.

Usage
-----
    python cyber_accumulation_model.py                 # default run, writes charts + CSVs
    python cyber_accumulation_model.py --years 50000   # more simulated years (smoother tail)
    python cyber_accumulation_model.py --mfa 0.95      # what if 95% of insureds had MFA?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

M = 1e6


# =============================================================================
# 1. ASSUMPTIONS  (every number the model uses lives here)
# =============================================================================
@dataclass
class Assumptions:
    n_policies: int = 5000
    n_years: int = 10000
    seed: int = 7

    # ---- portfolio composition ------------------------------------------------
    industries: dict = field(default_factory=lambda: {
        # share, frequency multiplier, severity multiplier
        "Professional services": (0.22, 1.00, 1.00),
        "Manufacturing":         (0.16, 1.15, 1.20),
        "Healthcare":            (0.12, 1.20, 1.40),
        "Financial services":    (0.12, 0.90, 1.30),
        "Retail":                (0.14, 1.00, 0.90),
        "Technology":            (0.12, 1.05, 1.10),
        "Education & public":    (0.12, 1.25, 0.80),
    })
    # revenue bands: share, (low, high) revenue in $, annual claim frequency, policy limit $, retention $
    revenue_bands: dict = field(default_factory=lambda: {
        "<$25M":      (0.45, (2e6, 25e6),    0.012, 1e6,  10e3),
        "$25M-$100M": (0.30, (25e6, 100e6),  0.025, 3e6,  25e3),
        "$100M-$1B":  (0.25, (100e6, 1e9),   0.057, 10e6, 100e3),
    })

    # ---- security controls (share of insureds that have each) -----------------
    p_mfa: float = 0.80
    p_edr: float = 0.60
    p_backups: float = 0.70           # tested, offline/immutable backups
    p_patching: float = 0.65          # critical patches within ~14 days

    # ---- attritional peril mix & severity -------------------------------------
    # share of claims, mean ground-up loss $ (at a $25M-revenue firm), lognormal sigma
    perils: dict = field(default_factory=lambda: {
        "BEC":          (0.33, 30e3,  1.2),
        "Funds transfer fraud": (0.25, 140e3, 1.3),
        "Ransomware":   (0.17, 450e3, 1.5),
        "Data breach":  (0.20, 250e3, 1.5),
        "Other":        (0.05, 80e3,  1.2),
    })
    severity_revenue_elasticity: float = 0.35   # severity scales with (revenue/$25M)^0.35

    # control effects: multiplier on frequency (f) or severity (s) of named perils
    mfa_effect: dict = field(default_factory=lambda: {"BEC": 0.55, "Funds transfer fraud": 0.70, "Ransomware": 0.60, "Data breach": 0.80})
    edr_effect: dict = field(default_factory=lambda: {"Ransomware": 0.55, "Data breach": 0.85})
    backup_sev_effect: dict = field(default_factory=lambda: {"Ransomware": 0.50})
    patch_effect: dict = field(default_factory=lambda: {"Ransomware": 0.80, "Data breach": 0.85})

    # ---- systemic: cloud providers ------------------------------------------
    cloud_share: dict = field(default_factory=lambda: {"Cloud A": 0.38, "Cloud B": 0.30, "Cloud C": 0.12, "On-prem/other": 0.20})
    cloud_outage_rate: float = 0.06       # per provider per year: major outage beyond waiting period
    outage_median_hours: float = 18.0
    outage_sigma: float = 0.7
    waiting_period_hours: float = 12.0
    p_dependent_bi_cover: float = 0.75    # share of insureds whose policy covers dependent BI / system failure
    net_income_plus_expense: float = 0.35 # covered BI loss as share of lost revenue (lost profit + extra expense)
    bi_sublimit: float = 0.50             # BI sublimit as share of policy limit

    # ---- systemic: software supply chain ---------------------------------------
    software_share: dict = field(default_factory=lambda: {"Software S1": 0.20, "Software S2": 0.15, "Software S3": 0.10,
                                                          "Software S4": 0.08, "Software S5": 0.05})
    software_compromise_rate: float = 0.02       # per vendor per year (a damaging, mass-exploited compromise)
    software_hit_beta: tuple = (1.0, 9.0)        # share of the vendor's users actually breached (mean 10%)
    software_loss_mean: float = 300e3
    software_loss_sigma: float = 1.3

    # ---- systemic: mass ransomware campaign ------------------------------------
    campaign_rate: float = 0.10                   # campaigns per year
    campaign_hit_rate: tuple = (0.001, 0.015)     # uniform range: share of exposed insureds hit
    campaign_unpatched_multiplier: float = 4.0    # unpatched insureds this many times likelier to be hit

    # ---- policy terms (v2) ---------------------------------------------------
    aggregate_limit: bool = True                  # policy limit is an annual aggregate across all events

    # ---- reinsurance & capital (v2) -------------------------------------------
    xl_retention: float = 20e6                    # cat excess-of-loss on each systemic event
    xl_limit: float = 40e6
    xl_reinstatements: int = 1                    # annual XL capacity = limit x (1 + reinstatements)
    xl_loading: float = 2.0                       # XL premium = expected recovery x loading
    qs_cession: float = 0.20                      # quota share of what is left after the XL
    qs_ceding_commission: float = 0.30
    expense_ratio: float = 0.30                   # acquisition + admin, % of gross premium

    # ---- pricing --------------------------------------------------------------
    target_loss_ratio: float = 0.60               # premium = total modeled AAL (attritional + cat) / target LR


# =============================================================================
# 2. PORTFOLIO
# =============================================================================
def build_portfolio(a: Assumptions, rng: np.random.Generator) -> pd.DataFrame:
    n = a.n_policies
    ind_names = list(a.industries)
    ind = rng.choice(ind_names, n, p=[a.industries[k][0] for k in ind_names])
    band_names = list(a.revenue_bands)
    band = rng.choice(band_names, n, p=[a.revenue_bands[k][0] for k in band_names])

    lo = np.array([a.revenue_bands[b][1][0] for b in band])
    hi = np.array([a.revenue_bands[b][1][1] for b in band])
    revenue = np.exp(rng.uniform(np.log(lo), np.log(hi)))   # log-uniform within band

    df = pd.DataFrame({
        "industry": ind, "band": band, "revenue": revenue,
        "base_freq": [a.revenue_bands[b][2] for b in band],
        "limit": [a.revenue_bands[b][3] for b in band],
        "retention": [a.revenue_bands[b][4] for b in band],
        "freq_mult": [a.industries[i][1] for i in ind],
        "sev_mult": [a.industries[i][2] for i in ind],
        "mfa": rng.random(n) < a.p_mfa,
        "edr": rng.random(n) < a.p_edr,
        "backups": rng.random(n) < a.p_backups,
        "patching": rng.random(n) < a.p_patching,
        "dep_bi_cover": rng.random(n) < a.p_dependent_bi_cover,
    })
    clouds = list(a.cloud_share)
    df["cloud"] = rng.choice(clouds, n, p=[a.cloud_share[c] for c in clouds])
    for s, share in a.software_share.items():         # an insured can use several products
        df[s] = rng.random(n) < share
    return df


def apply_terms(ground_up: np.ndarray, retention: np.ndarray, limit: np.ndarray) -> np.ndarray:
    return np.clip(ground_up - retention, 0, limit)


def lognormal_with_mean(rng, mean, sigma, size):
    mu = np.log(mean) - sigma**2 / 2
    return rng.lognormal(mu, sigma, size)


# =============================================================================
# 3. SIMULATION
# =============================================================================
SOURCES = ["BEC", "Funds transfer fraud", "Ransomware", "Data breach", "Other",
           "Cloud outage", "Software supply chain", "Ransomware campaign"]
SRC_ID = {s: i for i, s in enumerate(SOURCES)}


class _Ledger:
    """Collects every (year, policy, loss, source, occurrence) record before policy terms
    that span the whole year (the annual aggregate limit) are applied."""
    def __init__(self):
        self.y, self.p, self.l, self.s, self.o = [], [], [], [], []
        self.next_occ = 0
        self.occ_meta = []          # occurrence id -> (type, vendor, year, severity_param)

    def add(self, years, pols, losses, source, occ_ids):
        keep = losses > 0
        self.y.append(np.asarray(years)[keep]); self.p.append(np.asarray(pols)[keep])
        self.l.append(np.asarray(losses)[keep]); self.s.append(np.full(keep.sum(), SRC_ID[source], np.int8))
        self.o.append(np.asarray(occ_ids)[keep])

    def new_occ(self, kind, vendor, year, param):
        self.occ_meta.append((kind, vendor, year, param)); self.next_occ += 1
        return self.next_occ - 1

    def arrays(self):
        return [np.concatenate(a) for a in (self.y, self.p, self.l, self.s, self.o)]


def simulate(a: Assumptions, pf: pd.DataFrame | None = None):
    """Returns pf, by_source (annual insured loss per source, after aggregate limits),
    max_occ (largest occurrence per year), ev (systemic event log), occ (all occurrences)."""
    rng = np.random.default_rng(a.seed)
    if pf is None:
        pf = build_portfolio(a, rng)
    Y, n = a.n_years, len(pf)
    rev, ret, lim = pf.revenue.values, pf.retention.values, pf.limit.values
    sev_scale = pf.sev_mult.values * (rev / 25e6) ** a.severity_revenue_elasticity
    led = _Ledger()

    # ---- 3a. attritional: every claim is its own occurrence --------------------
    for peril, (share, mean, sigma) in a.perils.items():
        lam = pf.base_freq.values * pf.freq_mult.values * share
        sev_m = np.ones(n)
        for flag, eff in [("mfa", a.mfa_effect), ("edr", a.edr_effect), ("patching", a.patch_effect)]:
            if peril in eff:
                lam = lam * np.where(pf[flag].values, eff[peril], 1.0)
        if peril in a.backup_sev_effect:
            sev_m = np.where(pf.backups.values, a.backup_sev_effect[peril], 1.0)
        counts = rng.poisson(lam.sum(), Y)
        k = counts.sum()
        who = rng.choice(n, k, p=lam / lam.sum())
        gu = lognormal_with_mean(rng, mean, sigma, k) * sev_scale[who] * sev_m[who]
        loss = apply_terms(gu, ret[who], lim[who])
        occ = -1 - (np.arange(k) + sum(len(x) for x in led.l))   # negative ids = single-claim occurrences
        led.add(np.repeat(np.arange(Y), counts), who, loss, peril, occ)

    # ---- 3b. cloud outages --------------------------------------------------------
    for c in a.cloud_share:
        if c == "On-prem/other":
            continue
        dep = np.where((pf.cloud.values == c) & pf.dep_bi_cover.values)[0]
        for y in np.repeat(np.arange(Y), rng.poisson(a.cloud_outage_rate, Y)):
            hours = rng.lognormal(np.log(a.outage_median_hours), a.outage_sigma)
            excess = max(hours - a.waiting_period_hours, 0)
            if excess == 0:
                continue
            gu = rev[dep] / 8760 * excess * rng.beta(2, 5, len(dep)) * a.net_income_plus_expense
            loss = apply_terms(gu, ret[dep], lim[dep] * a.bi_sublimit)
            o = led.new_occ("Cloud outage", c, y, hours)
            led.add(np.full(len(dep), y), dep, loss, "Cloud outage", np.full(len(dep), o))

    # ---- 3c. software supply-chain compromise ------------------------------------
    for s in a.software_share:
        users = np.where(pf[s].values)[0]
        for y in np.repeat(np.arange(Y), rng.poisson(a.software_compromise_rate, Y)):
            frac = rng.beta(*a.software_hit_beta)
            hit = users[rng.random(len(users)) < frac]
            gu = lognormal_with_mean(rng, a.software_loss_mean, a.software_loss_sigma, len(hit)) * sev_scale[hit]
            o = led.new_occ("Software supply chain", s, y, frac)
            led.add(np.full(len(hit), y), hit, apply_terms(gu, ret[hit], lim[hit]), "Software supply chain", np.full(len(hit), o))

    # ---- 3d. mass ransomware campaign ---------------------------------------------
    rw_mean, rw_sigma = a.perils["Ransomware"][1], a.perils["Ransomware"][2]
    vuln = np.where(pf.patching.values, 1.0, a.campaign_unpatched_multiplier)
    vuln = vuln * np.where(pf.edr.values, a.edr_effect["Ransomware"], 1.0)
    backup_m = np.where(pf.backups.values, a.backup_sev_effect["Ransomware"], 1.0)
    for y in np.repeat(np.arange(Y), rng.poisson(a.campaign_rate, Y)):
        hit = np.where(rng.random(n) < np.minimum(rng.uniform(*a.campaign_hit_rate) * vuln, 1))[0]
        gu = lognormal_with_mean(rng, rw_mean, rw_sigma, len(hit)) * sev_scale[hit] * backup_m[hit]
        o = led.new_occ("Ransomware campaign", "Campaign", y, len(hit))
        led.add(np.full(len(hit), y), hit, apply_terms(gu, ret[hit], lim[hit]), "Ransomware campaign", np.full(len(hit), o))

    # ---- 3e. annual aggregate limit per policy (v2) -------------------------------
    yy, pp, ll, ss, oo = led.arrays()
    if a.aggregate_limit:
        key = yy.astype(np.int64) * n + pp
        uniq, inv = np.unique(key, return_inverse=True)
        tot = np.bincount(inv, weights=ll)
        cap = lim[(uniq % n)]
        scale = np.minimum(1.0, cap / np.maximum(tot, 1e-9))
        ll = ll * scale[inv]

    by_source = pd.DataFrame({src: np.bincount(yy[ss == i], weights=ll[ss == i], minlength=Y)
                              for i, src in enumerate(SOURCES)})
    # occurrences: systemic events (id >= 0) and single claims (id < 0)
    sysm = oo >= 0
    ev_loss = np.bincount(oo[sysm], weights=ll[sysm], minlength=led.next_occ)
    ev = pd.DataFrame(led.occ_meta, columns=["type", "vendor", "year", "severity_param"])
    ev["loss"] = ev_loss
    occ_year = np.concatenate([ev.year.values, yy[~sysm]])
    occ_loss = np.concatenate([ev_loss, ll[~sysm]])
    occ_sys = np.concatenate([np.ones(len(ev), bool), np.zeros((~sysm).sum(), bool)])
    max_occ = np.zeros(Y)
    np.maximum.at(max_occ, occ_year, occ_loss)
    occ = pd.DataFrame({"year": occ_year, "loss": occ_loss, "systemic": occ_sys})
    return pf, by_source, max_occ, ev, occ


# =============================================================================
# 4. METRICS
# =============================================================================
ATTRITIONAL = ["BEC", "Funds transfer fraud", "Ransomware", "Data breach", "Other"]
SYSTEMIC = ["Cloud outage", "Software supply chain", "Ransomware campaign"]
RETURN_PERIODS = [10, 25, 50, 100, 200, 250, 500]


def return_period_loss(x: np.ndarray, rp: float) -> float:
    return float(np.quantile(x, 1 - 1 / rp))


def tvar(x: np.ndarray, q: float) -> float:
    t = np.quantile(x, q)
    return float(x[x >= t].mean())


def summarize(a: Assumptions, pf, by_source, max_occ, ev, premium=None) -> dict:
    total = by_source.sum(axis=1).values
    att = by_source[ATTRITIONAL].sum(axis=1).values
    if premium is None:
        premium = total.mean() / a.target_loss_ratio
    tail_years = total >= np.quantile(total, 1 - 1 / 200)
    tail_mix = by_source[tail_years].mean() / total[tail_years].mean()
    vend_aal = ev.groupby("vendor").loss.sum() / a.n_years
    vend_tail = ev[ev.year.isin(np.where(tail_years)[0])].groupby("vendor").loss.sum() / tail_years.sum()

    return {
        "policies": len(pf),
        "total_limit_$M": pf.limit.sum() / M,
        "premium_$M": premium / M,
        "AAL_total_$M": total.mean() / M,
        "AAL_attritional_$M": att.mean() / M,
        "AAL_systemic_$M": (total - att).mean() / M,
        "AAL_by_source_$M": (by_source.mean() / M).round(3).to_dict(),
        "expected_loss_ratio": total.mean() / premium,
        "cat_load_share_of_premium": (total - att).mean() / premium,
        "avg_premium_per_policy_$": premium / len(pf),
        "premium_rate_on_line": premium / pf.limit.sum(),
        "AEP_$M": {rp: return_period_loss(total, rp) / M for rp in RETURN_PERIODS},
        "OEP_$M": {rp: return_period_loss(max_occ, rp) / M for rp in RETURN_PERIODS},
        "loss_ratio_at_RP": {rp: return_period_loss(total, rp) / premium for rp in RETURN_PERIODS},
        "TVaR_99.5_$M": tvar(total, 0.995) / M,
        "PML200_share_of_limit": return_period_loss(total, 200) / pf.limit.sum(),
        "tail_mix_1in200": tail_mix.round(3).to_dict(),
        "vendor_AAL_$M": (vend_aal / M).round(3).to_dict(),
        "vendor_tail_contribution_$M": (vend_tail / M).round(2).to_dict(),
        "dependents_by_cloud": pf.cloud.value_counts().to_dict(),
    }


def reinsurance_and_capital(a: Assumptions, by_source: pd.DataFrame, occ: pd.DataFrame, premium: float) -> dict:
    """Gross vs net view: per-event cat XL on systemic events, then a quota share, then capital.
    Capital = the 1-in-200 underwriting loss (99.5% VaR of the annual underwriting result)."""
    Y = len(by_source)
    gross = by_source.sum(axis=1).values
    sys_occ = occ[occ.systemic]
    rec_occ = np.clip(sys_occ.loss.values - a.xl_retention, 0, a.xl_limit)
    rec = np.bincount(sys_occ.year.values, weights=rec_occ, minlength=Y)
    rec = np.minimum(rec, a.xl_limit * (1 + a.xl_reinstatements))
    xl_prem = rec.mean() * a.xl_loading
    after_xl = gross - rec
    net_loss = after_xl * (1 - a.qs_cession)
    p_after_xl = premium - xl_prem
    net_prem = p_after_xl * (1 - a.qs_cession) + a.qs_ceding_commission * a.qs_cession * p_after_xl
    expenses = a.expense_ratio * premium
    res_gross = premium - expenses - gross
    res_net = net_prem - expenses - net_loss

    def view(result, losses, prem):
        cap = max(-np.quantile(result, 0.005), 1e-9)
        return {"premium_$M": prem / M, "AAL_$M": losses.mean() / M,
                "AEP_100_$M": return_period_loss(losses, 100) / M, "AEP_200_$M": return_period_loss(losses, 200) / M,
                "expected_UW_profit_$M": result.mean() / M, "required_capital_$M": cap / M,
                "return_on_capital": result.mean() / cap,
                "prob_UW_loss_year": float((result < 0).mean())}
    return {"gross": view(res_gross, gross, premium),
            "net": view(res_net, net_loss, net_prem),
            "xl": {"layer": f"${a.xl_limit/M:.0f}M xs ${a.xl_retention/M:.0f}M per systemic event",
                   "expected_recovery_$M": rec.mean() / M, "premium_$M": xl_prem / M,
                   "rate_on_line": xl_prem / a.xl_limit, "prob_layer_hit": float((rec > 0).mean())},
            "qs": {"cession": a.qs_cession, "ceding_commission": a.qs_ceding_commission},
            "_series": {"gross": gross, "net": net_loss}}


def stress_scenarios(a: Assumptions, pf: pd.DataFrame) -> dict:
    """Deterministic 'what if' events, the way a risk committee would ask for them."""
    rev, ret, lim = pf.revenue.values, pf.retention.values, pf.limit.values
    sev_scale = pf.sev_mult.values * (rev / 25e6) ** a.severity_revenue_elasticity
    out = {}
    for c, hours in [("Cloud A", 24), ("Cloud A", 72), ("Cloud B", 72)]:
        dep = (pf.cloud.values == c) & pf.dep_bi_cover.values
        gu = rev / 8760 * max(hours - a.waiting_period_hours, 0) * 2 / 7 * a.net_income_plus_expense  # mean beta(2,5)=2/7
        out[f"{c} outage, {hours}h"] = float(apply_terms(gu, ret, lim * a.bi_sublimit)[dep].sum() / M)
    for s, frac in [("Software S1", 0.30), ("Software S1", 0.60)]:
        users = pf[s].values
        gu = a.software_loss_mean * sev_scale
        out[f"{s} compromise, {int(frac*100)}% of users breached"] = float(frac * apply_terms(gu, ret, lim)[users].sum() / M)
    return out


# =============================================================================
# 5. CHARTS
# =============================================================================
def make_charts(a, by_source, max_occ, summary, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    total = by_source.sum(axis=1).values / M
    paths = []

    # EP curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for series, lab, col in [(total, "Aggregate (AEP)", "#2a6f97"), (max_occ / M, "Largest occurrence (OEP)", "#e07a5f")]:
        s = np.sort(series)[::-1]
        rp = a.n_years / np.arange(1, len(s) + 1)
        ax.plot(rp, s, label=lab, color=col, lw=2)
    for rp in (100, 200):
        ax.axvline(rp, color="gray", ls=":", lw=1)
        ax.text(rp, ax.get_ylim()[1] * 0.95, f" 1-in-{rp}", color="gray", fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(1, a.n_years)
    ax.set_xlabel("Return period (years, log scale)"); ax.set_ylabel("Insured loss ($M)")
    ax.set_title("Exceedance probability curves"); ax.legend(frameon=False)
    fig.tight_layout(); p = f"{outdir}/ep_curve.png"; fig.savefig(p, dpi=150); paths.append(p); plt.close(fig)

    # AAL vs tail mix
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    order = ATTRITIONAL + SYSTEMIC
    colors = ["#a8c5da"] * len(ATTRITIONAL) + ["#e07a5f"] * len(SYSTEMIC)
    aal = by_source[order].mean() / by_source.sum(axis=1).mean()
    tail = pd.Series(summary["tail_mix_1in200"])[order]
    for ax, data, title in [(axs[0], aal, "Share of average annual loss"), (axs[1], tail, "Share of loss in worst 1-in-200 years")]:
        ax.barh(order[::-1], data[order[::-1]] * 100, color=colors[::-1])
        ax.set_title(title); ax.set_xlabel("%")
    fig.suptitle("Attritional (blue) vs systemic (orange): who drives the average vs the tail", fontsize=11)
    fig.tight_layout(); p = f"{outdir}/aal_vs_tail.png"; fig.savefig(p, dpi=150); paths.append(p); plt.close(fig)

    # vendor concentration
    vt = pd.Series(summary["vendor_tail_contribution_$M"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(vt.index, vt.values, color="#3d405b")
    ax.set_xlabel("Average loss in 1-in-200+ years ($M)")
    ax.set_title("Which shared dependency drives tail losses")
    fig.tight_layout(); p = f"{outdir}/vendor_tail.png"; fig.savefig(p, dpi=150); paths.append(p); plt.close(fig)

    # annual loss distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(total, bins=120, color="#81b29a")
    ax.axvline(total.mean(), color="black", ls="--", lw=1, label=f"Mean ${total.mean():.1f}M")
    ax.axvline(summary["AEP_$M"][200], color="#e07a5f", lw=1.5, label=f"1-in-200 ${summary['AEP_$M'][200]:.1f}M")
    ax.set_xlabel("Annual insured loss ($M)"); ax.set_ylabel("Simulated years")
    ax.set_title("Distribution of annual portfolio loss"); ax.legend(frameon=False)
    fig.tight_layout(); p = f"{outdir}/loss_distribution.png"; fig.savefig(p, dpi=150); paths.append(p); plt.close(fig)
    return paths


def gross_net_chart(a, series, outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, col in [("gross", "#2a6f97"), ("net", "#81b29a")]:
        srt = np.sort(series[k])[::-1] / M
        ax.plot(a.n_years / np.arange(1, len(srt) + 1), srt, label=k.title(), color=col, lw=2)
    ax.set_xscale("log"); ax.set_xlim(1, a.n_years)
    ax.axvline(200, color="gray", ls=":", lw=1)
    ax.set_xlabel("Return period (years, log scale)"); ax.set_ylabel("Annual loss ($M)")
    ax.set_title("Aggregate loss before and after reinsurance"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{outdir}/gross_vs_net_ep.png", dpi=150); plt.close(fig)


# =============================================================================
# 6. CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, default=Assumptions.n_years)
    ap.add_argument("--policies", type=int, default=Assumptions.n_policies)
    ap.add_argument("--mfa", type=float, help="override MFA adoption")
    ap.add_argument("--seed", type=int, default=Assumptions.seed)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    a = Assumptions(n_years=args.years, n_policies=args.policies, seed=args.seed)
    if args.mfa is not None:
        a.p_mfa = args.mfa
    pf, by_source, max_occ, ev, occ = simulate(a)
    s = summarize(a, pf, by_source, max_occ, ev)
    s["stress_$M"] = stress_scenarios(a, pf)
    ri = reinsurance_and_capital(a, by_source, occ, s["premium_$M"] * M)
    series = ri.pop("_series")
    s["reinsurance_and_capital"] = ri

    # control sensitivity: same portfolio & seed, better controls, same premium
    better = Assumptions(**{**asdict(a), "p_mfa": 0.95, "p_edr": 0.85, "p_backups": 0.90, "p_patching": 0.85})
    pf_b, bs_b, mo_b, ev_b, _ = simulate(better)
    sb = summarize(better, pf_b, bs_b, mo_b, ev_b, premium=s["premium_$M"] * M)
    s["sensitivity_better_controls"] = {
        "AAL_total_$M": sb["AAL_total_$M"], "AEP_200_$M": sb["AEP_$M"][200],
        "expected_loss_ratio_same_premium": sb["expected_loss_ratio"]}

    print(json.dumps(s, indent=2, default=float))
    by_source.to_csv(f"{args.out}/annual_losses_by_source.csv", index_label="year")
    ev.to_csv(f"{args.out}/systemic_events.csv", index=False)
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(s, f, indent=2, default=float)
    make_charts(a, by_source, max_occ, s, args.out)
    gross_net_chart(a, series, args.out)


if __name__ == "__main__":
    main()
