"""McNemar's test on a Spark DataFrame (Reference §8.2, §8.18).

The distributed part is building the 2x2 paired table across potentially
billions of matched pairs. Each row is one subject with two 0/1 columns
(before, after). We ``groupBy(before, after).count()`` to get (a, b, c, d),
then the McNemar statistic is a scalar computation on the driver.

Run:  python mcnemar_test.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def mcnemar_spark(df, before_col: str, after_col: str) -> dict:
    """Compute McNemar's test from a Spark DataFrame of paired 0/1 columns."""
    agg = (df.groupBy(before_col, after_col)
             .agg(F.count(F.lit(1)).alias("n"))
             .collect())
    a = b = c = d = 0
    for r in agg:
        x = int(r[before_col]); y = int(r[after_col]); n = int(r["n"])
        if x == 1 and y == 1: a = n
        elif x == 1 and y == 0: b = n
        elif x == 0 and y == 1: c = n
        elif x == 0 and y == 0: d = n
    n_disc = b + c
    if n_disc == 0:
        return {"a": a, "b": b, "c": c, "d": d, "n_discordant": 0,
                "chi_square": 0.0, "p_value_asymptotic": 1.0,
                "p_value_exact": 1.0,
                "note": "all pairs concordant; test undefined"}
    x2 = (b - c) ** 2 / n_disc
    p_asym = float(stats.chi2.sf(x2, 1))
    k = min(b, c)
    p_exact = float(min(1.0, 2.0 * stats.binom.cdf(k, n_disc, 0.5)))
    or_hat = (b if b else 0.5) / (c if c else 0.5)
    return {"a": a, "b": b, "c": c, "d": d, "n_discordant": n_disc,
            "chi_square": x2, "df": 1,
            "p_value_asymptotic": p_asym,
            "p_value_exact": p_exact,
            "OR": or_hat,
            "method": "McNemar (asymptotic + exact) on Spark 2x2 aggregation"}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("mcnemar").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(7)
        rows = []
        for _ in range(2000):
            x = 1 if random.random() < 0.35 else 0
            p_pos = 0.90 if x == 1 else 0.15
            y = 1 if random.random() < p_pos else 0
            rows.append((x, y))
        df = spark.createDataFrame(rows, ["before", "after"])
        for k, v in mcnemar_spark(df, "before", "after").items():
            print(f"  {k:20s}: {v}")
    finally:
        spark.stop()
