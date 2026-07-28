"""Statistics for eval results, done properly.

Two things separate a credible eval report from a misleading one: an unbiased
pass@k estimator, and an interval that reflects the small sample sizes evals
actually run at. Both are pure functions here, unit-tested against known values.

pass@k
------
We report the *unbiased* pass@k estimator from Chen et al. (2021, "Evaluating
Large Language Models Trained on Code"). Given ``n`` independent attempts of
which ``c`` passed, the probability that at least one of a random ``k``-subset
passes is::

    pass@k = 1 - C(n - c, k) / C(n, k)

This is preferable to the naive "did any of my k runs pass" because it corrects
for the finite sample: with ``n = k`` it reduces to ``c > 0``, and with ``n > k``
it estimates what a fresh draw of ``k`` would yield. It requires ``n >= k``.

Wilson score interval
----------------------
For a pass *rate* (a binomial proportion) the textbook normal-approximation
("Wald") interval is wrong at the sample sizes evals use: near 0% or 100% it
produces intervals that fall outside ``[0, 1]`` and badly under-covers. The
Wilson score interval is the correct closed form — it stays in ``[0, 1]``,
degrades gracefully at the extremes, and is what you should quote next to a
pass rate. We report a 95% Wilson interval by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021).

    Args:
        n: total number of independent attempts.
        c: number of attempts that passed.
        k: the k in pass@k; must satisfy ``k <= n``.

    Returns:
        The estimated probability that at least one of a random k-subset of the
        n attempts passes, in ``[0, 1]``.
    """
    if k > n:
        raise ValueError(f"pass_at_k requires k <= n (got k={k}, n={n})")
    if not 0 <= c <= n:
        raise ValueError(f"pass_at_k requires 0 <= c <= n (got c={c}, n={n})")
    if n - c < k:
        # Every k-subset must contain at least one passing attempt.
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# z-score for a two-sided 95% interval.
_Z_95 = 1.959963984540054


def wilson_interval(
    successes: int, trials: int, z: float = _Z_95
) -> tuple[float, float]:
    """95% Wilson score confidence interval for a binomial proportion.

    Args:
        successes: number of passing trials.
        trials: total number of trials.
        z: standard-normal quantile (default 1.96 for 95%).

    Returns:
        ``(low, high)`` bounds in ``[0, 1]``. For ``trials == 0`` returns the
        vacuous ``(0.0, 1.0)``.
    """
    if trials == 0:
        return (0.0, 1.0)
    if successes < 0 or successes > trials:
        raise ValueError("successes must be in [0, trials]")
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2 * trials)) / denom
    margin = (
        z
        * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
        / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class RateStat:
    """A pass rate with its Wilson interval and pass@k, for one grouping."""

    label: str
    trials: int
    successes: int
    k: int

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.trials)

    @property
    def pass_at_k(self) -> float:
        """pass@k where possible, else the observed rate.

        The unbiased estimator needs ``k <= trials``. When a group aggregates
        many tasks (so ``trials`` is a multiple of ``k``) we treat every trial
        as an independent attempt and report the raw pass rate, which is the
        right aggregate quantity; the per-task ``k`` is carried for context.
        """
        if self.k and self.k <= self.trials and self.trials == self.k:
            return pass_at_k(self.trials, self.successes, self.k)
        return self.rate
