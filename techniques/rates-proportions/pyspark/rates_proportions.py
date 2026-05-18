"""Rates and proportions from a Spark DataFrame (Reference §1.8).

The distributed part is the *aggregation* -- counting cases and summing
person-time across a large cohort, optionally by stratum. Once you have the
event count and the denominator, the confidence intervals are tiny scalar
computations (done on the driver here, reusing the closed-form Wilson / exact
Poisson formulas).

Run:  python rates_proportions.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _wilson(x, n, conf=0.95):
    p = x / n
    z = stats.norm.ppf(0.5 + conf / 2)
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _poisson_ci(events, pt, conf=0.95):
    a = 1 - conf
    lo = 0.0 if events == 0 else stats.chi2.ppf(a / 2, 2 * events) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * events + 2) / 2
    return lo / pt, hi / pt


def prevalence_by_group(df, case_col: str, group_col: str):
    """Per-group prevalence with Wilson 95% CIs. ``case_col`` is 0/1."""
    agg = df.groupBy(group_col).agg(
        F.sum(case_col).alias("cases"), F.count(F.lit(1)).alias("n")
    ).collect()
    return [
        {group_col: r[group_col], "cases": r["cases"], "n": r["n"],
         "prevalence": r["cases"] / r["n"], "ci95": _wilson(r["cases"], r["n"])}
        for r in agg
    ]


def incidence_rate_by_group(df, event_col: str, time_col: str, group_col: str):
    """Per-group incidence rate (events / person-time) with exact Poisson 95% CIs."""
    agg = df.groupBy(group_col).agg(
        F.sum(event_col).alias("events"), F.sum(time_col).alias("person_time")
    ).collect()
    return [
        {group_col: r[group_col], "events": r["events"], "person_time": r["person_time"],
         "rate": r["events"] / r["person_time"], "ci95": _poisson_ci(r["events"], r["person_time"])}
        for r in agg
    ]


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("rates-proportions").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        # subject_id, arm, has_condition (0/1), got_event (0/1), follow_up_years
        rows = [
            (1, "A", 1, 0, 2.0), (2, "A", 0, 1, 5.0), (3, "A", 0, 0, 1.5), (4, "A", 1, 1, 5.0),
            (5, "B", 0, 0, 3.2), (6, "B", 1, 0, 5.0), (7, "B", 0, 1, 0.8), (8, "B", 0, 0, 4.1),
        ]
        df = spark.createDataFrame(rows, ["id", "arm", "has_condition", "got_event", "fu_years"])
        print("Prevalence by arm:")
        for r in prevalence_by_group(df, "has_condition", "arm"):
            print(" ", r)
        print("Incidence rate by arm (per person-year):")
        for r in incidence_rate_by_group(df, "got_event", "fu_years", "arm"):
            print(" ", r)
    finally:
        spark.stop()
