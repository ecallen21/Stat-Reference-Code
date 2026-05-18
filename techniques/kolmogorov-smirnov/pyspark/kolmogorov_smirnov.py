"""Kolmogorov-Smirnov test on Spark data (Reference §6.7).

For the ONE-sample case, Spark MLlib ships ``Statistics.kolmogorovSmirnovTest``
(against a normal distribution; you supply mean and SD). We expose that path
and fall back to a manual ECDF computation on the cluster otherwise.

For the TWO-sample case, the trick is the same as for Mann-Whitney: rank /
sort everything, walk the ECDFs, find the max gap. Implemented here via two
``approxQuantile``-driven empirical CDFs evaluated at a common grid.

Run:  python kolmogorov_smirnov.py
"""
from __future__ import annotations

import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy import stats


def one_sample_ks_spark_normal(df, col: str, mean: float, sd: float) -> dict:
    """One-sample KS test against N(mean, sd), via MLlib's built-in."""
    from pyspark.mllib.stat import Statistics
    rdd = df.select(col).rdd.map(lambda r: float(r[0]))
    res = Statistics.kolmogorovSmirnovTest(rdd, "norm", mean, sd)
    return {"D": float(res.statistic), "p_value": float(res.pValue),
            "method": "1-sample KS vs N(mu, sigma) [MLlib]"}


def two_sample_ks_spark(df, value_col: str, group_col: str,
                        grid_size: int = 1000) -> dict:
    """Two-sample KS via ECDFs evaluated at a quantile grid.

    Approximate; resolution improves with ``grid_size``. For exact two-sample KS
    on huge data, collect a stratified random sample and use the Python version.
    """
    counts = {r[group_col]: r["n"] for r in df.groupBy(group_col).count().withColumnRenamed("count", "n").collect()}
    if len(counts) != 2:
        raise ValueError("need exactly 2 groups")
    g1, g2 = sorted(counts.keys())
    # Build a shared grid via quantiles of the combined column
    grid = df.approxQuantile(value_col,
                             [i / grid_size for i in range(1, grid_size)],
                             0.0)
    # Per-group ECDF at each grid point
    df1 = df.filter(F.col(group_col) == g1).select(F.col(value_col).alias("v"))
    df2 = df.filter(F.col(group_col) == g2).select(F.col(value_col).alias("v"))
    n1 = counts[g1]; n2 = counts[g2]
    D = 0.0
    for t in grid:
        f1 = df1.filter(F.col("v") <= t).count() / n1
        f2 = df2.filter(F.col("v") <= t).count() / n2
        D = max(D, abs(f1 - f2))
    en = math.sqrt(n1 * n2 / (n1 + n2))
    p = float(stats.kstwobign.sf(en * D))
    return {"D": D, "n1": n1, "n2": n2, "p_value_asymptotic": p,
            "method": "2-sample KS (grid approximation)"}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("ks-test").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(5)
        rows = [("A", random.gauss(0, 1)) for _ in range(500)] + \
               [("B", random.gauss(0.4, 1)) for _ in range(450)]
        df = spark.createDataFrame(rows, ["arm", "value"])

        print("=== 1-sample KS vs N(0,1) on arm A only ===")
        dfA = df.filter(F.col("arm") == "A").select("value")
        try:
            print(one_sample_ks_spark_normal(dfA, "value", 0.0, 1.0))
        except Exception as exc:
            print(f"  (MLlib path unavailable: {exc})")

        print("\n=== 2-sample KS (A vs B) ===")
        print(two_sample_ks_spark(df, "value", "arm", grid_size=100))
    finally:
        spark.stop()
