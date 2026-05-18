"""Measures of central tendency on a Spark DataFrame (Reference §1.1).

Distributed versions of the "center" summaries: useful when the data are too
large to pull into a single machine. Mean / weighted mean / geometric / harmonic
are simple aggregations; the median is computed via ``approxQuantile`` (exact
median of a huge column is expensive, so Spark offers an approximate one).

Run:  python central_tendency.py
"""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def central_tendency(df, col: str, weight_col: str | None = None, relative_error: float = 0.001):
    """Return a dict of central-tendency measures for ``col`` in a Spark DataFrame."""
    aggs = [
        F.mean(col).alias("arithmetic_mean"),
        F.exp(F.mean(F.log(col))).alias("geometric_mean"),       # requires col > 0
        (F.count(col) / F.sum(F.lit(1.0) / F.col(col))).alias("harmonic_mean"),
        F.count(col).alias("n"),
    ]
    if weight_col is not None:
        aggs.append((F.sum(F.col(col) * F.col(weight_col)) / F.sum(weight_col)).alias("weighted_mean"))
    row = df.agg(*aggs).first().asDict()

    # Mode: value with the highest frequency.
    mode_row = (df.groupBy(col).count().orderBy(F.desc("count"), F.asc(col)).first())
    row["mode"] = mode_row[col] if mode_row else None

    # Median (approximate): the 0.5 quantile.
    med = df.approxQuantile(col, [0.5], relative_error)
    row["median_approx"] = med[0] if med else None
    return row


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("central-tendency").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        rows = [(2,), (4,), (4,), (4,), (5,), (5,), (7,), (9,), (100,)]
        df = spark.createDataFrame(rows, ["x"]).withColumn("w", F.monotonically_increasing_id() + 1)
        result = central_tendency(df, "x", weight_col="w")
        for k, v in result.items():
            print(f"{k:20s}: {v}")
    finally:
        spark.stop()
