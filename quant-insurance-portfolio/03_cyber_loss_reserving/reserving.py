"""
Cyber loss reserving: chain ladder, Bornhuetter-Ferguson and a bootstrap reserve distribution.

The data is a simulated cyber book (accident years 2016-2025) with a known "true" ultimate for
each year, including a ransomware spike in 2020-2021, so every method can be scored against
the truth. Methods:

  * Chain ladder (paid and incurred): volume-weighted age-to-age factors
  * Bornhuetter-Ferguson: blends an expected loss ratio prior with the chain-ladder pattern
  * ODP bootstrap (England & Verrall): resamples Pearson residuals of the paid triangle to get a
    distribution of the total reserve, not just a point estimate

    python reserving.py
"""
import json, pathlib
import numpy as np
import pandas as pd

OUT = pathlib.Path("output"); OUT.mkdir(exist_ok=True)
AY = list(range(2016, 2026))
N = len(AY)
SEED = 11

# --------------------------------------------------------------------------------------
# 1. Simulate the book
# --------------------------------------------------------------------------------------
PREMIUM = np.array([20, 26, 34, 45, 58, 70, 82, 90, 96, 100], float) * 1e6     # fast-growing line
TRUE_ELR = np.array([0.55, 0.58, 0.62, 0.70, 0.95, 0.88, 0.60, 0.55, 0.57, 0.60])  # 2020-21 ransomware wave
PAID_PATTERN = np.array([0.22, 0.52, 0.73, 0.86, 0.93, 0.965, 0.985, 0.995, 1.0, 1.0])
INC_PATTERN = np.array([0.50, 0.78, 0.90, 0.96, 0.985, 0.995, 1.0, 1.0, 1.0, 1.0])
PRIOR_ELR = 0.62           # what pricing expected for every year (BF's prior)


def simulate(seed=SEED):
    rng = np.random.default_rng(seed)
    ult = PREMIUM * TRUE_ELR * rng.lognormal(-0.01, 0.12, N)      # realized ultimates differ from the ELR
    paid = np.full((N, N), np.nan); inc = np.full((N, N), np.nan)
    for i in range(N):
        inc_paid = np.diff(np.r_[0, PAID_PATTERN]) * ult[i] * rng.lognormal(-0.02, 0.2, N)
        cum = np.cumsum(inc_paid)
        cum = cum / cum[-1] * ult[i]                                   # pays out exactly the ultimate
        case = (INC_PATTERN * ult[i] - PAID_PATTERN * ult[i]) * rng.lognormal(0, 0.15, N)
        for j in range(N - i):                                         # only the observed diagonal
            paid[i, j] = cum[j]
            inc[i, j] = cum[j] + max(case[j], 0)
    return ult, pd.DataFrame(paid, index=AY, columns=range(1, N + 1)), pd.DataFrame(inc, index=AY, columns=range(1, N + 1))


# --------------------------------------------------------------------------------------
# 2. Methods
# --------------------------------------------------------------------------------------
def ata_factors(tri: np.ndarray) -> np.ndarray:
    """Volume-weighted age-to-age factors f_j = sum C[i,j+1] / sum C[i,j] over rows with both."""
    f = np.ones(tri.shape[1] - 1)
    for j in range(tri.shape[1] - 1):
        rows = ~np.isnan(tri[:, j + 1])
        f[j] = tri[rows, j + 1].sum() / tri[rows, j].sum() if rows.any() else 1.0
    return f


def latest(tri: np.ndarray):
    idx = np.array([np.max(np.where(~np.isnan(r))[0]) for r in tri])
    return tri[np.arange(len(tri)), idx], idx


def chain_ladder(tri: np.ndarray, tail=1.0):
    f = ata_factors(tri)
    cdf = np.r_[np.cumprod(f[::-1])[::-1], 1.0] * tail      # to-ultimate factor from each age
    cur, idx = latest(tri)
    return cur * cdf[idx], f, cdf


def bornhuetter_ferguson(tri, premium, prior_elr, cdf):
    cur, idx = latest(tri)
    pct_unreported = 1 - 1 / cdf[idx]
    return cur + premium * prior_elr * pct_unreported


def complete_triangle(tri, f):
    """Fill the lower triangle with chain-ladder projections."""
    full = tri.copy()
    for i in range(len(full)):
        for j in range(1, full.shape[1]):
            if np.isnan(full[i, j]):
                full[i, j] = full[i, j - 1] * f[j - 1]
    return full


def odp_bootstrap(paid: np.ndarray, n_sims=5000, seed=1):
    """England & Verrall over-dispersed Poisson bootstrap of the paid chain ladder."""
    rng = np.random.default_rng(seed)
    n = len(paid)
    f = ata_factors(paid)
    cur, idx = latest(paid)
    # back-fit the observed cumulative triangle from the latest diagonal using the factors
    fitted_cum = np.full_like(paid, np.nan)
    for i in range(n):
        fitted_cum[i, idx[i]] = cur[i]
        for j in range(idx[i] - 1, -1, -1):
            fitted_cum[i, j] = fitted_cum[i, j + 1] / f[j]
    inc_obs = np.diff(np.c_[np.zeros(n), paid], axis=1)
    inc_fit = np.diff(np.c_[np.zeros(n), fitted_cum], axis=1)
    obs = ~np.isnan(paid)
    use = obs & (inc_fit > 0)                                 # cells with no expected payment carry no residual
    r = (inc_obs[use] - inc_fit[use]) / np.sqrt(inc_fit[use])
    k = use.sum(); p = 2 * n - 1
    phi = np.sum(r ** 2) / (k - p)
    r_adj = r * np.sqrt(k / (k - p))                          # degrees-of-freedom adjustment
    reserves = np.empty(n_sims)
    for s in range(n_sims):
        rs = rng.choice(r_adj, k, replace=True)
        inc_star = np.full_like(paid, np.nan)
        inc_star[obs] = inc_fit[obs]
        inc_star[use] = inc_fit[use] + rs * np.sqrt(inc_fit[use])
        cum_star = np.cumsum(np.nan_to_num(inc_star), axis=1); cum_star[~obs] = np.nan
        f_star = ata_factors(cum_star)
        full = complete_triangle(cum_star, f_star)
        inc_future = np.diff(np.c_[np.zeros(n), full], axis=1)[~obs]
        inc_future = np.maximum(inc_future, 1e-6)
        # process variance: gamma with mean m and variance phi * m
        reserves[s] = rng.gamma(inc_future / phi, phi).sum()
    return reserves, phi


# --------------------------------------------------------------------------------------
# 3. Run and score
# --------------------------------------------------------------------------------------
def main():
    true_ult, paid_df, inc_df = simulate()
    paid, inc = paid_df.values, inc_df.values
    paid_df.to_csv(OUT / "paid_triangle.csv"); inc_df.to_csv(OUT / "incurred_triangle.csv")

    cl_paid, f_paid, cdf_paid = chain_ladder(paid)
    cl_inc, f_inc, cdf_inc = chain_ladder(inc)
    bf_paid = bornhuetter_ferguson(paid, PREMIUM, PRIOR_ELR, cdf_paid)
    bf_inc = bornhuetter_ferguson(inc, PREMIUM, PRIOR_ELR, cdf_inc)
    paid_to_date, idx = latest(paid)

    res = pd.DataFrame({
        "premium": PREMIUM, "paid_to_date": paid_to_date, "age": idx + 1,
        "CL_paid": cl_paid, "CL_incurred": cl_inc, "BF_paid": bf_paid, "BF_incurred": bf_inc,
        "true_ultimate": true_ult}, index=AY)
    methods = ["CL_paid", "CL_incurred", "BF_paid", "BF_incurred"]
    for m in methods:
        res[f"{m}_err%"] = 100 * (res[m] / res.true_ultimate - 1)
    res.to_csv(OUT / "ultimates_by_method.csv")

    true_reserve = (true_ult - paid_to_date).sum()
    reserve = {m: float((res[m] - paid_to_date).sum()) for m in methods}
    recent = res.index >= 2023
    mape_recent = {m: float(res.loc[recent, f"{m}_err%"].abs().mean()) for m in methods}
    mape_old = {m: float(res.loc[~recent, f"{m}_err%"].abs().mean()) for m in methods}

    boot, phi = odp_bootstrap(paid)
    pct = {f"p{q}": float(np.percentile(boot, q)) for q in (5, 25, 50, 75, 95, 99.5)}

    summary = {
        "true_total_reserve_$M": true_reserve / 1e6,
        "estimated_reserve_$M": {m: v / 1e6 for m, v in reserve.items()},
        "reserve_error_%": {m: 100 * (v / true_reserve - 1) for m, v in reserve.items()},
        "mean_abs_ult_error_%_AY2023_2025": mape_recent,
        "mean_abs_ult_error_%_AY2016_2022": mape_old,
        "paid_age_to_age_factors": [round(x, 3) for x in f_paid],
        "bootstrap": {"mean_$M": boot.mean() / 1e6, "cv": boot.std() / boot.mean(),
                      **{k: v / 1e6 for k, v in pct.items()},
                      "true_reserve_percentile": float((boot < true_reserve).mean() * 100),
                      "odp_dispersion_phi": phi},
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2))
    show = res[methods + ["true_ultimate"]] / 1e6
    show.insert(0, "age", res.age)
    print(show.round(2).to_string())
    charts(res, methods, boot, true_reserve, f_paid, f_inc)


def charts(res, methods, boot, true_reserve, f_paid, f_inc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(res))
    cols = {"CL_paid": "#e07a5f", "CL_incurred": "#f2cc8f", "BF_paid": "#2a6f97", "BF_incurred": "#81b29a"}
    for m in methods:
        axs[0].plot(x, res[f"{m}_err%"], "o-", color=cols[m], label=m.replace("_", " "), ms=4)
    axs[0].axhline(0, color="black", lw=0.8)
    axs[0].set_xticks(x); axs[0].set_xticklabels(res.index, rotation=45)
    axs[0].set_ylabel("Error vs true ultimate (%)"); axs[0].set_title("Ultimate loss error by accident year")
    axs[0].legend(frameon=False, ncol=2, fontsize=8)
    axs[1].hist(boot / 1e6, bins=60, color="#a8c5da")
    axs[1].axvline(true_reserve / 1e6, color="#e07a5f", lw=2, label=f"True reserve ${true_reserve/1e6:.0f}M")
    axs[1].axvline(np.percentile(boot, 50) / 1e6, color="black", ls="--", lw=1, label="Bootstrap median")
    axs[1].axvline(np.percentile(boot, 99.5) / 1e6, color="gray", ls=":", lw=1, label="99.5th percentile")
    axs[1].set_xlabel("Total unpaid reserve ($M)"); axs[1].set_title("ODP bootstrap reserve distribution (paid)")
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "reserving_results.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
