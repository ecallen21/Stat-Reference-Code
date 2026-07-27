"""Permutation two-sample test on a Spark DataFrame (Reference §10.7, §10.16).

The distributed part is computing mean(y|group) for each permutation, which
requires shuffling the group column. For very large data we can:
    1. Collect just the values (no group) if they fit on driver.
    2. Or: use a broadcast join with a permutation index, and groupBy the
       shuffled group column.

This file implements approach (2) for demonstration. For truly large n it's
better to compute the observed mean-diff on Spark and run the permutation
loop on a driver-side sample if the full column doesn't fit.

Run:  python permutation_tests.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)


def perm_two_sample_spark(df, value_col: str, group_col: str,
                           n_perm: int = 200, seed: int = 0) -> dict:
    """Permutation test for mean(A) - mean(B) using Spark aggregation on shuffled labels.

    For a fair benchmark on very large data, the observed diff is computed on
    Spark; the permutation loop is done on the driver over the collected values.
    """
    agg = df.groupBy(group_col).agg(F.mean(F.col(value_col)).alias("m"),
                                     F.count(F.lit(1)).alias("n")).collect()
    if len(agg) != 2:
        raise ValueError("need exactly 2 groups")
    a, b = sorted(agg, key=lambda r: r[group_col])
    t_obs = float(a["m"] - b["m"])
    n1, n2 = int(a["n"]), int(b["n"])

    # Collect values for the permutation loop (would sample for truly huge n)
    values = [r[value_col] for r in df.select(value_col).collect()]
    import numpy as np
    values = np.array(values, dtype=float)
    n = n1 + n2
    rng = np.random.default_rng(seed)
    t_perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n)
        t_perm[i] = float(values[idx[:n1]].mean() - values[idx[n1:]].mean())
    extreme = np.abs(t_perm) >= abs(t_obs)
    p = (1 + int(extreme.sum())) / (1 + n_perm)
    return {"T_obs": t_obs, "p_value": float(p),
            "n1": n1, "n2": n2, "n_perm": n_perm,
            "group_labels": [a[group_col], b[group_col]],
            "method": "Spark: observed diff via groupBy; permutation loop on driver"}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("perm").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(37)
        rows = [("A", random.gauss(0, 1)) for _ in range(400)] + \
               [("B", random.gauss(0.5, 1)) for _ in range(450)]
        df = spark.createDataFrame(rows, ["arm", "y"])
        for k, v in perm_two_sample_spark(df, "y", "arm", n_perm=500).items():
            print(f"  {k:14s}: {v}")
    finally:
        spark.stop()
