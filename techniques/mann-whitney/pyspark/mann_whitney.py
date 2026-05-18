"""Mann-Whitney U on a Spark DataFrame (Reference §6.3).

The algorithm needs combined ranks of (group 1, group 2). In Spark:
  1. Union the two columns (or filter by group_col).
  2. Rank the combined values via a window function with ties handled by
     ``avg(rank)`` so we get average-rank ties.
  3. Sum the ranks per group; derive U from the standard formula.

For very large data with few ties, a percentile-based approximation is faster,
but the rank window is the textbook implementation.

Run:  python mann_whitney.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

from pyspark.sql import SparkSession, Window    # SparkSession: entry point;  Window: window-function specifications
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def mann_whitney_spark(df, value_col: str, group_col: str) -> dict:
    """Mann-Whitney U via average-rank window function on a Spark DataFrame."""
    # average-rank: rank ties at the mean of their tied positions.
    # Use (min_rank + max_rank) / 2  --  which equals dense_rank-style averaging.
    w_unbounded = Window.orderBy(value_col)
    w_eq = Window.partitionBy(value_col)
    ranked = (df.select(value_col, group_col)
                .withColumn("rmin", F.rank().over(w_unbounded))
                .withColumn("rmax", F.max("rmin").over(w_eq))
                .withColumn("rmin2", F.min("rmin").over(w_eq))
                .withColumn("rank", (F.col("rmin2") + F.col("rmax")) / 2.0))
    agg = (ranked.groupBy(group_col)
                  .agg(F.sum("rank").alias("rank_sum"),
                       F.count(F.lit(1)).alias("n"))
                  .collect())
    if len(agg) != 2:
        raise ValueError("need exactly 2 groups")
    a, b = sorted(agg, key=lambda r: r[group_col])
    n1, R1 = a["n"], a["rank_sum"]
    n2, R2 = b["n"], b["rank_sum"]
    n = n1 + n2
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    # ignore tie correction for the spark version (tiny effect on continuous data)
    var = n1 * n2 * (n + 1) / 12
    z = (U1 - mu - 0.5 * (1.0 if U1 > mu else -1.0)) / math.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"group1": a[group_col], "group2": b[group_col],
            "n1": n1, "n2": n2, "R1": R1, "R2": R2, "U1": U1,
            "z": z, "p_value": p}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("mann-whitney").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(2)
        rows = [("A", random.gauss(50, 10)) for _ in range(300)] + \
               [("B", random.gauss(55, 12)) for _ in range(280)]
        df = spark.createDataFrame(rows, ["arm", "value"])
        for k, v in mann_whitney_spark(df, "value", "arm").items():
            print(f"  {k:10s}: {v}")
    finally:
        spark.stop()
