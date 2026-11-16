"""
Cyber frequency-severity pricing model.

1. Pull policies and claims from SQLite.
2. Fit a Poisson GLM for claim frequency (offset = log exposure) and a Gamma GLM for claim
   severity on accident years 2023-2024.
3. Check the fitted relativities against the known truth, and cross-check the frequency fit
   against scikit-learn.
4. Validate on 2025 (out of time): lift chart, double-lift vs the current rating plan, Gini.
5. Write the indicated rating plan: base rate x relativities.

    python generate_data.py && python pricing_model.py
"""
import json, pathlib, sqlite3
import numpy as np
import pandas as pd
from glm import fit_glm
from generate_data import INDUSTRIES, TRUE

OUT = pathlib.Path("output"); OUT.mkdir(exist_ok=True)
BASE_INDUSTRY = "Professional services"
IND = [i for i in INDUSTRIES if i != BASE_INDUSTRY]


def load():
    con = sqlite3.connect("cyber_book.db")
    pol = pd.read_sql("""
        SELECT p.*, COALESCE(c.n_claims, 0) AS n_claims, COALESCE(c.incurred, 0) AS incurred
        FROM policies p LEFT JOIN (SELECT policy_id, COUNT(*) n_claims, SUM(amount) incurred
                                   FROM claims GROUP BY policy_id) c USING (policy_id)""", con)
    clm = pd.read_sql("SELECT c.*, p.industry, p.revenue, p.mfa, p.edr, p.backups, p.patching, p.prior_claim "
                      "FROM claims c JOIN policies p USING (policy_id)", con)
    return pol, clm


def design(df, terms):
    cols = {"intercept": np.ones(len(df))}
    for i in IND:
        cols[f"industry: {i}"] = (df.industry == i).astype(float).values
    cols["log revenue / $25M"] = np.log(df.revenue.values / 25e6)
    for t in terms:
        cols[t] = df[t].astype(float).values if t != "year_trend" else (df.year.values - 2023).astype(float)
    X = pd.DataFrame(cols)
    return X.values, list(X.columns)


FREQ_TERMS = ["mfa", "edr", "patching", "backups", "prior_claim"]
SEV_TERMS = ["backups", "edr", "mfa", "year_trend"]


def truth_table():
    """True relativities used to generate the data, in the GLM's terms."""
    f = {f"industry: {i}": INDUSTRIES[i][1] for i in IND}
    f.update({"log revenue / $25M": TRUE["rev_freq_elasticity"], "mfa": TRUE["mfa_freq"], "edr": TRUE["edr_freq"],
              "patching": TRUE["patch_freq"], "backups": 1.0, "prior_claim": TRUE["prior_claim_freq"]})
    s = {f"industry: {i}": INDUSTRIES[i][2] for i in IND}
    s.update({"log revenue / $25M": TRUE["rev_sev_elasticity"], "backups": TRUE["backup_sev"], "edr": TRUE["edr_sev"],
              "mfa": 1.0, "year_trend": 1 + TRUE["trend"]})
    return f, s


def gini(actual, pred, weight):
    """Normalized-by-nothing Gini from the Lorenz curve of actual loss, ordering by predicted loss cost."""
    o = np.argsort(pred)
    w = np.cumsum(weight[o]) / weight.sum()
    a = np.cumsum(actual[o]) / actual.sum()
    return 1 - 2 * np.trapezoid(a, w)


def main():
    pol, clm = load()
    train, test = pol[pol.year < 2025], pol[pol.year == 2025]
    ctrain = clm[clm.year < 2025]

    # ---------------- frequency: Poisson GLM with log-exposure offset --------------------
    Xf, nf = design(train, FREQ_TERMS)
    freq = fit_glm(Xf, train.n_claims.values, "poisson", nf, offset=np.log(train.exposure.values))
    # ---------------- severity: Gamma GLM on individual claims --------------------------
    Xs, ns = design(ctrain, SEV_TERMS)
    sev = fit_glm(Xs, ctrain.amount.values, "gamma", ns)

    # ---------------- cross-check against scikit-learn ----------------------------------
    from sklearn.linear_model import PoissonRegressor
    sk = PoissonRegressor(alpha=0, max_iter=1000, tol=1e-10).fit(
        Xf[:, 1:], train.n_claims.values / train.exposure.values, sample_weight=train.exposure.values)
    max_diff = float(np.max(np.abs(np.r_[sk.intercept_, sk.coef_] - freq.coef)))

    # ---------------- fitted vs true ----------------------------------------------------
    tf, ts = truth_table()
    ft, st = freq.table(), sev.table()
    def compare(t, truth, cont="log revenue / $25M"):
        rows = []
        for k in t.index[1:]:
            fitted = t.loc[k, "coef"] if k == cont else t.loc[k, "relativity"]
            lo = t.loc[k, "coef"] - 1.96 * t.loc[k, "se"] if k == cont else t.loc[k, "rel_low_95"]
            hi = t.loc[k, "coef"] + 1.96 * t.loc[k, "se"] if k == cont else t.loc[k, "rel_high_95"]
            rows.append({"term": k, "true": truth[k], "fitted": fitted, "low_95": lo, "high_95": hi,
                         "truth_in_CI": lo <= truth[k] <= hi})
        return pd.DataFrame(rows)
    cf, cs = compare(ft, tf), compare(st, ts)
    cf.to_csv(OUT / "frequency_fitted_vs_true.csv", index=False)
    cs.to_csv(OUT / "severity_fitted_vs_true.csv", index=False)
    ft.to_csv(OUT / "frequency_glm_coefficients.csv"); st.to_csv(OUT / "severity_glm_coefficients.csv")

    # ---------------- out-of-time validation on 2025 ------------------------------------
    Xf_t, _ = design(test, FREQ_TERMS)
    Xs_t, _ = design(test, SEV_TERMS)
    pred_freq = freq.predict(Xf_t)                  # per exposure-year
    pred_sev = sev.predict(Xs_t)                    # includes trend to 2025
    pure = pred_freq * pred_sev                     # expected loss per exposure-year
    exp_ = test.exposure.values
    actual = test.incurred.values
    cur_rate = test.premium.values / exp_           # current plan's rate per exposure-year

    # lift chart: deciles of predicted pure premium
    dec = pd.qcut(pure, 10, labels=False)
    lift = pd.DataFrame({"decile": dec + 1, "pred": pure * exp_, "actual": actual, "exp": exp_}).groupby("decile").sum()
    lift["pred_loss_cost"] = lift.pred / lift.exp
    lift["actual_loss_cost"] = lift.actual / lift.exp
    lift.to_csv(OUT / "lift_chart.csv")

    # double lift: sort by GLM / current-plan ratio; current plan's loss ratio per bucket
    ratio = (pure * 1 / 0.62) / cur_rate             # GLM premium at the same target LR / current premium
    q = pd.qcut(ratio, 5, labels=False)
    dl = pd.DataFrame({"q": q + 1, "actual": actual, "cur_prem": test.premium.values,
                       "glm_prem": pure * exp_ / 0.62}).groupby("q").sum()
    dl["current_plan_LR"] = dl.actual / dl.cur_prem
    dl["glm_plan_LR"] = dl.actual / dl.glm_prem
    dl.to_csv(OUT / "double_lift.csv")

    g_glm = gini(actual, pure, exp_)
    g_cur = gini(actual, cur_rate, exp_)

    # ---------------- indicated rating plan ----------------------------------------------
    base_rate = float(np.exp(ft.loc["intercept", "coef"] + st.loc["intercept", "coef"] + 2 * st.loc["year_trend", "coef"]) / 0.62)
    plan = []
    for k in [f"industry: {i}" for i in IND]:
        plan.append({"factor": k, "relativity": ft.loc[k, "relativity"] * st.loc[k, "relativity"]})
    for k in ["mfa", "edr", "patching", "backups", "prior_claim"]:
        rf = ft.loc[k, "relativity"] if k in ft.index else 1.0
        rs = st.loc[k, "relativity"] if k in st.index else 1.0
        plan.append({"factor": f"has {k}" if k != "prior_claim" else "prior claim", "relativity": rf * rs})
    plan = pd.DataFrame(plan)
    plan.to_csv(OUT / "rating_plan_relativities.csv", index=False)
    rev_exp = float(ft.loc["log revenue / $25M", "coef"] + st.loc["log revenue / $25M", "coef"])

    summary = {
        "train_policies": len(train), "train_claims": int(train.n_claims.sum()), "test_policies": len(test),
        "frequency_deviance_explained": 1 - freq.deviance / freq.null_deviance,
        "severity_dispersion": sev.dispersion,
        "max_coef_diff_vs_sklearn": max_diff,
        "true_relativities_inside_95CI": f"{int(cf.truth_in_CI.sum() + cs.truth_in_CI.sum())} of {len(cf) + len(cs)}",
        "gini_glm": g_glm, "gini_current_plan": g_cur,
        "test_LR_current_plan": float(actual.sum() / test.premium.sum()),
        "base_rate_per_policy_2025_$25M_prof_services_no_controls": base_rate,
        "revenue_exponent_premium": rev_exp,
        "double_lift_current_plan_LR_by_quintile": dl.current_plan_LR.round(3).tolist(),
        "double_lift_glm_plan_LR_by_quintile": dl.glm_plan_LR.round(3).tolist(),
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    pd.set_option("display.width", 160)
    print(json.dumps(summary, indent=2))
    print("\nFrequency: fitted vs true\n", cf.round(3).to_string(index=False))
    print("\nSeverity: fitted vs true\n", cs.round(3).to_string(index=False))
    print("\nRating plan\n", plan.round(3).to_string(index=False))
    charts(cf, cs, lift, dl)


def charts(cf, cs, lift, dl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, c, title in [(axs[0], cf, "Frequency relativities"), (axs[1], cs, "Severity relativities")]:
        c = c[c.term != "log revenue / $25M"].iloc[::-1]
        y = np.arange(len(c))
        ax.errorbar(c.fitted, y, xerr=[c.fitted - c.low_95, c.high_95 - c.fitted], fmt="o", color="#2a6f97",
                    capsize=3, label="Fitted (95% CI)")
        ax.scatter(c.true, y, marker="x", color="#e07a5f", zorder=3, label="True")
        ax.axvline(1, color="gray", lw=0.8, ls=":")
        ax.set_yticks(y); ax.set_yticklabels(c.term); ax.set_title(title); ax.set_xlabel("Relativity")
    axs[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "relativities_fitted_vs_true.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(lift.index, lift.pred_loss_cost, "o-", color="#2a6f97", label="Predicted")
    axs[0].plot(lift.index, lift.actual_loss_cost, "s--", color="#e07a5f", label="Actual")
    axs[0].set_xlabel("Decile of predicted loss cost (2025 holdout)"); axs[0].set_ylabel("Loss cost per policy-year ($)")
    axs[0].set_title("Lift chart: GLM separates low from high risk"); axs[0].legend(frameon=False)
    x = np.arange(1, len(dl) + 1)
    axs[1].bar(x - 0.2, dl.current_plan_LR * 100, 0.4, color="#bbbbbb", label="Current plan")
    axs[1].bar(x + 0.2, dl.glm_plan_LR * 100, 0.4, color="#2a6f97", label="GLM plan")
    axs[1].set_xlabel("Quintile of GLM premium / current premium"); axs[1].set_ylabel("Loss ratio (%)")
    axs[1].axhline(62, color="black", ls=":", lw=1, label="Target 62%")
    axs[1].set_title("Double lift: current plan misprices the extremes"); axs[1].legend(frameon=False, loc="upper left")
    axs[1].set_ylim(0, 135)
    fig.tight_layout(); fig.savefig(OUT / "lift_and_double_lift.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
