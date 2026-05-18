"""Spearman rank correlation on a Spark DataFrame (Reference §4.2).

Spearman's rho is Pearson's r on the *ranks* of the data. In Spark we rank
each column with a window function (``percent_rank`` -- or ``row_number`` /
``dense_rank`` if you want exact average ties), then compute ``F.corr`` on the
two rank columns. Single shuffle for each ``orderBy``; cheap.

Run:  python spearman_rank_correlation.py
"""
from __future__ import annotations

import math

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from scipy import stats


def spearman_test_spark(df, col1: str, col2: str, conf: float = 0.95) -> dict:
    """Spearman rho via rank() in Spark then F.corr.

    Note: ``rank()`` here is the window-function rank (1-based, ties get the
    same rank, which differs slightly from average-rank ties used by scipy/R
    on heavily-tied data). For continuous data with few/no ties the difference
    is negligible.
    """
    w1 = Window.orderBy(col1); w2 = Window.orderBy(col2)
    ranked = (df
              .withColumn("rx", F.rank().over(w1))
              .withColumn("ry", F.rank().over(w2)))
    row = ranked.agg(F.corr("rx", "ry").alias("rho"),
                     F.count(F.lit(1)).alias("n")).first()
    r, n = row["rho"], row["n"]
    df_t = n - 2
    t = r * math.sqrt(df_t / (1 - r * r)) if abs(r) < 1 else float("inf")
    p = float(2 * stats.t.sf(abs(t), df_t))
    z = math.atanh(r); se = 1 / math.sqrt(n - 3)
    zc = stats.norm.ppf(0.5 + conf / 2)
    return {"rho": r, "n": n, "t": t, "df": df_t, "p_value": p,
            "ci_lower": math.tanh(z - zc * se),
            "ci_upper": math.tanh(z + zc * se)}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("spearman").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(6)
        rows = []
        for _ in range(200):
            x = random.gauss(0, 1); rows.append((x, math.exp(x) + random.gauss(0, 0.3)))
        df = spark.createDataFrame(rows, ["x", "y"])
        for k, v in spearman_test_spark(df, "x", "y").items():
            print(f"  {k:10s}: {v}")
    finally:
        spark.stop()
