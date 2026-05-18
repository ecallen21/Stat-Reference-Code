"""Measures of dispersion on a Spark DataFrame (Reference §1.2).

Variance / SD are built-in aggregations. Range is max - min. IQR uses
``approxQuantile``. MAD needs the column median first, then the median of the
absolute deviations -- two passes of ``approxQuantile``.

Run:  python dispersion.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)


def dispersion(df, col: str, relative_error: float = 0.001):
    stats = df.agg(
        F.min(col).alias("min"),
        F.max(col).alias("max"),
        F.var_samp(col).alias("variance"),     # n-1 denominator
        F.stddev_samp(col).alias("std"),
        F.mean(col).alias("mean"),
    ).first().asDict()
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] else None

    q1, med, q3 = df.approxQuantile(col, [0.25, 0.5, 0.75], relative_error)
    stats["IQR"] = q3 - q1

    dev = df.select(F.abs(F.col(col) - F.lit(med)).alias("d"))
    mad = dev.approxQuantile("d", [0.5], relative_error)[0]
    stats["MAD"] = mad
    stats["MAD_scaled_sigma"] = mad * 1.4826
    return stats


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("dispersion").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.createDataFrame([(v,) for v in [4, 8, 6, 5, 3, 9, 7, 11, 6, 100]], ["x"])
        for k, v in dispersion(df, "x").items():
            print(f"{k:18s}: {v}")
    finally:
        spark.stop()
