"""pass@k and Wilson CI against hand-computed known values."""

from __future__ import annotations

import math

import pytest

from gauntlet.metrics import RateStat, pass_at_k, wilson_interval


def test_pass_at_k_all_pass() -> None:
    assert pass_at_k(n=5, c=5, k=1) == 1.0


def test_pass_at_k_none_pass() -> None:
    assert pass_at_k(n=5, c=0, k=3) == 0.0


def test_pass_at_k_k_equals_n_reduces_to_any_pass() -> None:
    # With k == n, pass@k is 1 iff at least one attempt passed.
    assert pass_at_k(n=4, c=1, k=4) == 1.0
    assert pass_at_k(n=4, c=0, k=4) == 0.0


def test_pass_at_k_known_value() -> None:
    # n=5, c=1, k=1: probability a single random draw is the one passing = 1/5.
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    # n=4, c=2, k=2: 1 - C(2,2)/C(4,2) = 1 - 1/6 = 5/6.
    assert pass_at_k(4, 2, 2) == pytest.approx(5 / 6)
    # n=3, c=1, k=2: 1 - C(2,2)/C(3,2) = 1 - 1/3 = 2/3.
    assert pass_at_k(3, 1, 2) == pytest.approx(2 / 3)


def test_pass_at_k_requires_k_le_n() -> None:
    with pytest.raises(ValueError):
        pass_at_k(n=2, c=1, k=3)


def test_pass_at_k_rejects_bad_c() -> None:
    with pytest.raises(ValueError):
        pass_at_k(n=2, c=3, k=1)


def test_wilson_interval_known_value() -> None:
    # 8/10 at 95%: Wilson center/half-width are standard textbook values.
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4901, abs=1e-3)
    assert high == pytest.approx(0.9430, abs=1e-3)


def test_wilson_interval_half_is_symmetric_around_half() -> None:
    low, high = wilson_interval(5, 10)
    assert (low + high) / 2 == pytest.approx(0.5, abs=1e-9)


def test_wilson_interval_extremes_stay_in_unit() -> None:
    low, high = wilson_interval(0, 10)
    assert low == 0.0  # clamped; never negative like the Wald interval
    assert 0.0 < high < 0.35
    low2, high2 = wilson_interval(10, 10)
    assert high2 == pytest.approx(1.0)
    assert 0.65 < low2 < 1.0


def test_wilson_interval_zero_trials_is_vacuous() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_ratestat_rate_and_pass_at_k() -> None:
    stat = RateStat(label="cap", trials=3, successes=1, k=3)
    assert stat.rate == pytest.approx(1 / 3)
    # trials == k, so the unbiased estimator applies: c>0 -> 1.0 at k=n.
    assert stat.pass_at_k == 1.0
    lo, hi = stat.interval
    assert 0.0 <= lo <= hi <= 1.0


def test_ratestat_aggregate_uses_raw_rate() -> None:
    # trials > k: the aggregate reports the observed pass rate.
    stat = RateStat(label="all", trials=30, successes=24, k=3)
    assert stat.pass_at_k == pytest.approx(0.8)
    assert not math.isnan(stat.pass_at_k)
