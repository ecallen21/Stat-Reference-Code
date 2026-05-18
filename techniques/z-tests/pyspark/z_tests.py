"""Two-proportion z-test on a Spark DataFrame (Reference §3.7).

Use this when the data live as one row per subject with a 0/1 outcome flag and
a group column (e.g. arm A vs B). The cluster computes per-group counts and
sizes; the closed-form pooled-SE z-test happens on the driver. This is the
right pattern for A/B tests with very large datasets.

Run:  python z_tests.py
"""
from __future__ import annotations

import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy import stats


def two_proportion_z_spark(df, outcome_col: str, group_col: str,
                           alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """Two-proportion z-test from group counts on a Spark DataFrame.

    ``outcome_col`` should be 0/1 (success = 1). Exactly two distinct values
    of ``group_col`` are expected.
    """
    agg = (df.groupBy(group_col)
             .agg(F.sum(outcome_col).alias("x"), F.count(F.lit(1)).alias("n"))
             .collect())
    if len(agg) != 2:
        raise ValueError(f"need exactly 2 groups, found {len(agg)}")
    a, b = sorted(agg, key=lambda r: r[group_col])
    x1, n1 = a["x"], a["n"]
    x2, n2 = b["x"], b["n"]
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
    elif alternative == "greater":
        p = float(stats.norm.sf(z))
    else:
        p = float(stats.norm.cdf(z))
    zc = stats.norm.ppf(0.5 + conf / 2)
    se_u = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {"group1": a[group_col], "group2": b[group_col],
            "x1": x1, "n1": n1, "p1": p1, "x2": x2, "n2": n2, "p2": p2,
            "diff": p1 - p2, "z": z, "p_value": p,
            "ci_lower": (p1 - p2) - zc * se_u, "ci_upper": (p1 - p2) + zc * se_u}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("z-tests").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(1)
        rows = [("A", 1 if random.random() < 0.42 else 0) for _ in range(2_000)] + \
               [("B", 1 if random.random() < 0.30 else 0) for _ in range(2_000)]
        df = spark.createDataFrame(rows, ["arm", "conv"])
        print("Two-proportion z-test (conversion rate by arm):")
        for k, v in two_proportion_z_spark(df, "conv", "arm").items():
            print(f"  {k:10s}: {v}")
    finally:
        spark.stop()
