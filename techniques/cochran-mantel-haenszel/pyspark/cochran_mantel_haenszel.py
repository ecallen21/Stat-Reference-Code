"""Cochran-Mantel-Haenszel test on a Spark DataFrame (Reference §8.3, §8.16).

The distributed part is building the K 2x2 tables when the raw data has one
row per subject with (stratum, exposure, outcome) columns and K is potentially
large (many age-sex-region cells). ``groupBy(stratum, exposure, outcome).count()``
aggregates to at most 4K rows, which we ``collect()`` and finish on the driver.

Run:  python cochran_mantel_haenszel.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def cmh_spark(df, stratum_col: str, exposure_col: str, outcome_col: str) -> dict:
    """CMH on a Spark DataFrame; each row is one subject.

    ``exposure_col`` and ``outcome_col`` must be 0/1. ``stratum_col`` is any hashable.
    """
    agg = (df.groupBy(stratum_col, exposure_col, outcome_col)
             .agg(F.count(F.lit(1)).alias("n"))
             .collect())
    # rebuild K 2x2 tables on the driver
    strata = {}
    for r in agg:
        s = r[stratum_col]; e = int(r[exposure_col]); o = int(r[outcome_col]); n = int(r["n"])
        tbl = strata.setdefault(s, [[0, 0], [0, 0]])
        # exposure=1 -> row 0; outcome=1 -> col 0
        i = 0 if e == 1 else 1
        j = 0 if o == 1 else 1
        tbl[i][j] = n
    # CMH numerator/denominator
    numer = denom = 0.0
    R = S = 0.0
    for tbl in strata.values():
        a, b = tbl[0]; c, d = tbl[1]
        N = a + b + c + d
        if N <= 1: continue
        n1 = a + b; n0 = c + d; m1 = a + c; m0 = b + d
        e_a = n1 * m1 / N
        v_a = (n1 * n0 * m1 * m0) / (N * N * (N - 1))
        numer += a - e_a; denom += v_a
        R += a * d / N; S += b * c / N
    if denom == 0 or S == 0:
        return {"K_strata": len(strata), "chi_square": 0.0, "p_value": 1.0,
                "OR_MH": None, "note": "zero variance or zero denominator"}
    x2 = numer * numer / denom
    return {"K_strata": len(strata),
            "chi_square": x2, "df": 1,
            "p_value": float(stats.chi2.sf(x2, 1)),
            "OR_MH": R / S,
            "method": "CMH (asymptotic) + MH OR on Spark stratified aggregation"}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("cmh").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(1)
        # 3 strata; treatment increases outcome odds ~2x in each
        rows = []
        for s in ["young", "middle", "old"]:
            base = {"young": 0.25, "middle": 0.35, "old": 0.55}[s]
            for _ in range(600):
                e = 1 if random.random() < 0.5 else 0
                p = min(0.95, base * (2.0 if e == 1 else 1.0))
                o = 1 if random.random() < p else 0
                rows.append((s, e, o))
        df = spark.createDataFrame(rows, ["age_group", "treated", "event"])
        for k, v in cmh_spark(df, "age_group", "treated", "event").items():
            print(f"  {k:14s}: {v}")
    finally:
        spark.stop()
