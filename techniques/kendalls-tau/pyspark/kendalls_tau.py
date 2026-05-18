"""Kendall's tau on a Spark DataFrame (Reference §4.3).

Kendall's tau requires counting concordant and discordant *pairs*, which is
naturally O(n^2). On Spark we self-join the DataFrame to itself with an
``i < j`` predicate and aggregate the signs of the (delta_x, delta_y) products.

This is fine for moderate ``n``; for very large ``n`` use the O(n log n) merge-
sort algorithm (Knight 1966) on a single machine after sampling, or pull a
random sub-sample to the driver. Spark MLlib does NOT ship Kendall in
``Correlation.corr`` (only Pearson and Spearman) -- this is the standard
workaround.

Run:  python kendalls_tau.py
"""
from __future__ import annotations

import math

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from scipy import stats


def kendall_tau_b_spark(df, col1: str, col2: str) -> dict:
    """Kendall's tau-b via a self-join on the DataFrame.

    Adds a row id, then joins each row with all higher-id rows. For each pair
    we classify concordant / discordant / tied. Quadratic in ``n``; use only
    for ``n`` up to ~10^4 (a 100M-pair self-join is ~10K rows on each side).
    """
    indexed = df.select(F.col(col1).alias("x"), F.col(col2).alias("y")) \
                .withColumn("id", F.monotonically_increasing_id())
    a = indexed.alias("a"); b = indexed.alias("b")
    joined = a.join(b, F.col("a.id") < F.col("b.id"))
    counts = joined.agg(
        F.sum(F.when((F.col("a.x") == F.col("b.x")) & (F.col("a.y") == F.col("b.y")), 1)
              .otherwise(0)).alias("Txy"),
        F.sum(F.when((F.col("a.x") == F.col("b.x")) & (F.col("a.y") != F.col("b.y")), 1)
              .otherwise(0)).alias("Tx"),
        F.sum(F.when((F.col("a.x") != F.col("b.x")) & (F.col("a.y") == F.col("b.y")), 1)
              .otherwise(0)).alias("Ty"),
        F.sum(F.when(((F.col("a.x") - F.col("b.x")) * (F.col("a.y") - F.col("b.y"))) > 0,
                     1).otherwise(0)).alias("C"),
        F.sum(F.when(((F.col("a.x") - F.col("b.x")) * (F.col("a.y") - F.col("b.y"))) < 0,
                     1).otherwise(0)).alias("D"),
    ).first().asDict()
    n = df.count()
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - counts["Tx"]) * (n0 - counts["Ty"]))
    tau = (counts["C"] - counts["D"]) / denom if denom > 0 else float("nan")
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (counts["C"] - counts["D"]) / math.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"tau_b": tau, **counts, "n": n, "z": z, "p_value": p}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("kendall").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(7)
        rows = [(random.gauss(0, 1),) for _ in range(120)]
        rows = [(x, 0.5 * x + math.sqrt(0.75) * random.gauss(0, 1)) for (x,) in rows]
        df = spark.createDataFrame(rows, ["x", "y"])
        for k, v in kendall_tau_b_spark(df, "x", "y").items():
            print(f"  {k:10s}: {v}")
    finally:
        spark.stop()
