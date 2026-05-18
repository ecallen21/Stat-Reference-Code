"""Summary statistics for rates and proportions (Reference §1.8).

Measures of how often an event occurs in a population:

  - prevalence            : proportion WITH a condition at a point in time (cross-sectional)
  - incidence proportion  : "risk" = proportion DEVELOPING the condition over a period
  - incidence rate        : events per person-time at risk (handles variable follow-up)
  - person-time           : sum of individual at-risk follow-up durations

Confidence intervals:
  for a proportion p = x/n
    - Wald (normal approx)    : p +/- z * sqrt(p(1-p)/n)        (poor near 0/1 or small n)
    - Wilson score            : recommended default
    - Clopper-Pearson (exact) : guaranteed >= nominal coverage; uses the Beta distribution
  for a rate lambda = events / person_time
    - exact Poisson CI        : via the chi-squared / Gamma relationship
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import NamedTuple, Sequence    # stdlib: type hints (Sequence = indexable iterable; NamedTuple = tuple with named fields)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


class CI(NamedTuple):
    """Confidence interval. Unpacks like a tuple: ``lo, hi = ci_wilson(x, n)``."""
    lower: float
    upper: float


# --------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------
def prevalence(cases: int, population: int) -> float:
    """Point prevalence = cases / population (cross-sectional)."""
    return cases / population


def incidence_proportion(new_cases: int, at_risk_at_start: int) -> float:
    """Cumulative incidence ("risk") = new_cases / at_risk_at_start over the follow-up period."""
    return new_cases / at_risk_at_start


def person_time(follow_up_times: Sequence[float]) -> float:
    """Total person-time = sum of individual at-risk follow-up durations.

    ``follow_up_times`` is a sequence with one entry per subject (e.g. years observed).
    """
    return float(sum(follow_up_times))


def incidence_rate(events: int, person_time_total: float) -> float:
    """Incidence rate = events / total person-time (e.g. per person-year)."""
    return events / person_time_total


# --------------------------------------------------------------------------
# Confidence intervals for a proportion
# --------------------------------------------------------------------------
# Conventions for the CI functions below:
#   x    -- number of successes (count of the event of interest)
#   n    -- sample size (total trials)
#   conf -- confidence level (e.g. 0.95 for a 95% CI)
# All return a CI named tuple with fields ``lower`` and ``upper``, clipped to
# [0, 1]. Unpacks like a tuple, so ``lo, hi = ci_wilson(x, n)`` still works.

def ci_wald(x: int, n: int, conf: float = 0.95):
    """Wald (normal-approximation) CI for a proportion. Poor near 0/1 and for small n."""
    p = x / n
    z = stats.norm.ppf(0.5 + conf / 2)
    half = z * math.sqrt(p * (1 - p) / n)
    return CI(lower=max(0.0, p - half), upper=min(1.0, p + half))


def ci_wilson(x: int, n: int, conf: float = 0.95):
    """Wilson score CI for a proportion -- the recommended general-purpose default."""
    p = x / n
    z = stats.norm.ppf(0.5 + conf / 2)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return CI(lower=max(0.0, center - half), upper=min(1.0, center + half))


def ci_clopper_pearson(x: int, n: int, conf: float = 0.95):
    """Exact (Clopper-Pearson) CI via Beta-distribution quantiles. Conservative but guaranteed coverage."""
    alpha = 1 - conf
    lower = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x, n - x + 1)
    upper = 1.0 if x == n else stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    return CI(lower=lower, upper=upper)


# --------------------------------------------------------------------------
# Confidence interval for a rate (exact Poisson)
# --------------------------------------------------------------------------
def ci_poisson_rate(events: int, person_time_total: float, conf: float = 0.95):
    """Exact Poisson (Garwood) CI for an incidence rate.

    Parameters
    ----------
    events : observed event count.
    person_time_total : denominator (sum of at-risk follow-up).
    conf : confidence level.

    Returns a CI named tuple with fields ``lower`` and ``upper`` -- bounds on
    the underlying rate ``lambda``. Unpacks like a tuple.
    """
    alpha = 1 - conf
    lo_count = 0.0 if events == 0 else stats.chi2.ppf(alpha / 2, 2 * events) / 2
    hi_count = stats.chi2.ppf(1 - alpha / 2, 2 * events + 2) / 2
    return CI(lower=lo_count / person_time_total,
              upper=hi_count / person_time_total)


def library_versions(x: int, n: int):
    from statsmodels.stats.proportion import proportion_confint

    return {
        "Wald (statsmodels)": proportion_confint(x, n, method="normal"),
        "Wilson (statsmodels)": proportion_confint(x, n, method="wilson"),
        "Clopper-Pearson (statsmodels)": proportion_confint(x, n, method="beta"),
        "Agresti-Coull (statsmodels)": proportion_confint(x, n, method="agresti_coull"),
    }


if __name__ == "__main__":
    print("=== Prevalence ===")
    print("120 cases in a population of 5000:", prevalence(120, 5000))

    print("\n=== Incidence ===")
    follow = [2.0, 5.0, 1.5, 5.0, 3.2, 5.0, 0.8, 4.1]  # person-years for 8 subjects
    pt = person_time(follow)
    print("person-time (years)    :", pt)
    print("incidence rate (3 events / py):", incidence_rate(3, pt), "per person-year")
    print("incidence proportion (3 / 8 at risk):", incidence_proportion(3, 8))

    print("\n=== 95% CIs for a proportion  x=8, n=100  (p = 0.08) ===")
    for name, fn in [("Wald", ci_wald), ("Wilson", ci_wilson), ("Clopper-Pearson", ci_clopper_pearson)]:
        lo, hi = fn(8, 100)
        print(f"{name:18s}: ({lo:.4f}, {hi:.4f})")

    print(f"\n=== 95% exact Poisson CI for a rate  3 events / {pt} person-years ===")
    lo, hi = ci_poisson_rate(3, pt)
    print(f"rate = {3 / pt:.4f}   CI = ({lo:.4f}, {hi:.4f}) per person-year")

    print("\n--- library (statsmodels.proportion_confint) ---")
    for k, v in library_versions(8, 100).items():
        print(f"{k:32s}: ({v[0]:.4f}, {v[1]:.4f})")
