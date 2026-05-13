"""Pearson correlation on a Spark DataFrame (Reference §4.1).

Spark has a built-in ``F.corr(col1, col2)`` aggregation, plus
``Statistics.corr`` / ``Correlation.corr`` in MLlib for full correlation
matrices. The closed-form Fisher-z CI runs on the driver from the scalar
``r`` and the total count.

Run:  python pearson_correlation.py
"""
from __future__ import annotations

import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy import stats


def pearson_test_spark(df, col1: str, col2: str, conf: float = 0.95) -> dict:
    """Pearson r + t-test + Fisher-z CI from a Spark DataFrame."""
    row = df.agg(F.corr(col1, col2).alias("r"),
                 F.count(F.lit(1)).alias("n")).first()
    r, n = row["r"], row["n"]
    df_t = n - 2
    t = r * math.sqrt(df_t / (1 - r * r)) if abs(r) < 1 else float("inf")
    p = float(2 * stats.t.sf(abs(t), df_t))
    z = math.atanh(r); se = 1 / math.sqrt(n - 3)
    zc = stats.norm.ppf(0.5 + conf / 2)
    return {"r": r, "n": n, "t": t, "df": df_t, "p_value": p,
            "ci_lower": math.tanh(z - zc * se),
            "ci_upper": math.tanh(z + zc * se)}


def correlation_matrix_spark(df, cols: list) -> list:
    """Per-pair Pearson correlations among ``cols``. Returns a list of (col_i, col_j, r)."""
    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = df.agg(F.corr(cols[i], cols[j])).first()[0]
            out.append((cols[i], cols[j], r))
    return out


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("pearson").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(5)
        rho = 0.6
        rows = [(random.gauss(0, 1), 0) for _ in range(200)]
        rows = [(x, rho * x + math.sqrt(1 - rho * rho) * random.gauss(0, 1)) for x, _ in rows]
        df = spark.createDataFrame(rows, ["x", "y"])
        print("Pearson:")
        for k, v in pearson_test_spark(df, "x", "y").items():
            print(f"  {k:10s}: {v}")
    finally:
        spark.stop()
