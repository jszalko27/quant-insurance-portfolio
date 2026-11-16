"""
Generate a synthetic cyber insurance book (policies + claims) with KNOWN true risk relativities,
and store it in a SQLite database. Because the truth is known, the GLM's ability to recover it
can be checked directly.

    python generate_data.py            # writes cyber_book.db
"""
import sqlite3
import numpy as np
import pandas as pd

SEED = 2026
N_POLICIES_PER_YEAR = 60000
YEARS = [2023, 2024, 2025]

INDUSTRIES = {  # share, true frequency relativity, true severity relativity
    "Professional services": (0.22, 1.00, 1.00),
    "Manufacturing":         (0.16, 1.20, 1.25),
    "Healthcare":            (0.12, 1.30, 1.45),
    "Financial services":    (0.12, 0.85, 1.30),
    "Retail":                (0.14, 1.05, 0.90),
    "Technology":            (0.12, 1.10, 1.10),
    "Education & public":    (0.12, 1.35, 0.80),
}
PERILS = {  # share of claims, relative severity
    "BEC": (0.33, 0.25), "Funds transfer fraud": (0.27, 0.90),
    "Data breach": (0.23, 1.30), "Ransomware": (0.17, 2.00),
}
TRUE = {
    "base_freq": 0.012,          # annual claim frequency for a $25M professional-services firm with no controls
    "rev_freq_elasticity": 0.45, # frequency scales with (revenue / $25M) ^ 0.45
    "mfa_freq": 0.60, "edr_freq": 0.75, "patch_freq": 0.85, "prior_claim_freq": 1.60,
    "base_sev": 120e3, "rev_sev_elasticity": 0.30,
    "backup_sev": 0.65, "edr_sev": 0.85,
    "gamma_shape": 0.8,          # claim-size CV = 1/sqrt(0.8) ~ 1.12
    "trend": 0.05,               # 5% annual severity trend
}


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    pols, claims = [], []
    pid = 0
    ind_names = list(INDUSTRIES)
    for yr in YEARS:
        n = N_POLICIES_PER_YEAR
        ind = rng.choice(ind_names, n, p=[INDUSTRIES[i][0] for i in ind_names])
        revenue = np.exp(rng.uniform(np.log(2e6), np.log(1e9), n))
        size = revenue / 25e6
        # larger firms are likelier to have controls (realistic correlation the GLM must untangle)
        p_ctrl = lambda base: np.clip(base + 0.08 * np.log10(size + 0.1), 0.05, 0.98)
        mfa = rng.random(n) < p_ctrl(0.75)
        edr = rng.random(n) < p_ctrl(0.55)
        backups = rng.random(n) < p_ctrl(0.65)
        patch = rng.random(n) < p_ctrl(0.60)
        prior = rng.random(n) < 0.08
        exposure = np.where(rng.random(n) < 0.85, 1.0, rng.uniform(0.25, 1.0, n))
        fr = np.array([INDUSTRIES[i][1] for i in ind])
        sr = np.array([INDUSTRIES[i][2] for i in ind])
        lam = (TRUE["base_freq"] * fr * size ** TRUE["rev_freq_elasticity"]
               * np.where(mfa, TRUE["mfa_freq"], 1) * np.where(edr, TRUE["edr_freq"], 1)
               * np.where(patch, TRUE["patch_freq"], 1) * np.where(prior, TRUE["prior_claim_freq"], 1))
        mu_sev = (TRUE["base_sev"] * sr * size ** TRUE["rev_sev_elasticity"]
                  * np.where(backups, TRUE["backup_sev"], 1) * np.where(edr, TRUE["edr_sev"], 1)
                  * (1 + TRUE["trend"]) ** (yr - YEARS[0]))
        # the CURRENT rating plan: priced on revenue and industry only (no credit for controls)
        crude = size ** 0.55 * (0.6 * fr + 0.4) * exposure * 1.05 ** (yr - YEARS[0])
        premium = crude * (lam * mu_sev * exposure).sum() / crude.sum() / 0.62   # book priced to a ~62% loss ratio
        counts = rng.poisson(lam * exposure)
        ids = np.arange(pid, pid + n); pid += n
        pols.append(pd.DataFrame({"policy_id": ids, "year": yr, "industry": ind, "revenue": revenue.round(0),
                                  "mfa": mfa.astype(int), "edr": edr.astype(int), "backups": backups.astype(int),
                                  "patching": patch.astype(int), "prior_claim": prior.astype(int),
                                  "exposure": exposure.round(4), "premium": premium.round(2)}))
        k = counts.sum()
        who = np.repeat(np.arange(n), counts)
        perils, mix, pf = list(PERILS), np.array([v[0] for v in PERILS.values()]), np.array([v[1] for v in PERILS.values()])
        pi = rng.choice(len(perils), k, p=mix)
        peril = np.array(perils)[pi]
        sev_factor = pf[pi] / (mix * pf).sum()       # peril mix changes the shape, not the overall mean
        amt = rng.gamma(TRUE["gamma_shape"], mu_sev[who] * sev_factor / TRUE["gamma_shape"])
        claims.append(pd.DataFrame({"policy_id": ids[who], "year": yr, "peril": peril, "amount": amt.round(2)}))
    pol = pd.concat(pols, ignore_index=True)
    clm = pd.concat(claims, ignore_index=True)
    clm.insert(0, "claim_id", np.arange(len(clm)))
    return pol, clm


if __name__ == "__main__":
    pol, clm = generate()
    con = sqlite3.connect("cyber_book.db")
    pol.to_sql("policies", con, if_exists="replace", index=False)
    clm.to_sql("claims", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_claims_policy ON claims(policy_id)")
    con.commit(); con.close()
    print(f"policies: {len(pol):,}  claims: {len(clm):,}  "
          f"frequency: {len(clm)/pol.exposure.sum():.3%}  avg claim: ${clm.amount.mean():,.0f}  "
          f"loss ratio: {clm.amount.sum()/pol.premium.sum():.1%}")
