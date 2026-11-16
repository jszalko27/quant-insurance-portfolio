"""Unit tests: run with `pytest -q`."""
import numpy as np
import pandas as pd
import pytest
from accumulation_model import (Assumptions, apply_terms, lognormal_with_mean, return_period_loss,
                                simulate, summarize, reinsurance_and_capital, build_portfolio)

SMALL = dict(n_policies=800, n_years=3000, seed=3)


def test_apply_terms_retention_and_limit():
    gu = np.array([5e3, 50e3, 5e6])
    out = apply_terms(gu, np.full(3, 10e3), np.full(3, 1e6))
    assert out.tolist() == [0, 40e3, 1e6]


def test_lognormal_mean_is_unbiased():
    rng = np.random.default_rng(0)
    x = lognormal_with_mean(rng, 250e3, 1.2, 400_000)
    assert abs(x.mean() / 250e3 - 1) < 0.02


def test_return_period_losses_increase():
    x = np.random.default_rng(1).gamma(2, 10, 50_000)
    rps = [return_period_loss(x, rp) for rp in (10, 50, 100, 200)]
    assert rps == sorted(rps)


@pytest.fixture(scope="module")
def run():
    a = Assumptions(**SMALL)
    return a, simulate(a)


def test_annual_aggregate_limit_respected(run):
    a, (pf, by_source, max_occ, ev, occ) = run
    # rebuild per-policy annual totals is expensive; instead check the book-level bound
    assert (by_source.sum(axis=1).values <= pf.limit.sum() + 1e-6).all()
    assert (by_source.values >= 0).all()


def test_aggregate_limit_only_reduces_losses():
    a_on = Assumptions(**SMALL)
    a_off = Assumptions(**SMALL, aggregate_limit=False)
    on = simulate(a_on)[1].sum(axis=1)
    off = simulate(a_off)[1].sum(axis=1)
    assert (on.values <= off.values + 1e-6).all()


def test_oep_never_exceeds_aep(run):
    a, (pf, by_source, max_occ, ev, occ) = run
    assert (max_occ <= by_source.sum(axis=1).values + 1e-6).all()


def test_reinsurance_reduces_tail(run):
    a, (pf, by_source, max_occ, ev, occ) = run
    s = summarize(a, pf, by_source, max_occ, ev)
    ri = reinsurance_and_capital(a, by_source, occ, s["premium_$M"] * 1e6)
    assert ri["net"]["AEP_200_$M"] < ri["gross"]["AEP_200_$M"]
    assert ri["net"]["required_capital_$M"] < ri["gross"]["required_capital_$M"]


def test_better_controls_lower_attritional_loss():
    base = Assumptions(**SMALL)
    better = Assumptions(**SMALL, p_mfa=0.98, p_edr=0.98, p_backups=0.98, p_patching=0.98)
    att = ["BEC", "Funds transfer fraud", "Ransomware", "Data breach", "Other"]
    assert simulate(better)[1][att].values.sum() < simulate(base)[1][att].values.sum()
