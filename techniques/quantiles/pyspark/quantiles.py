"""Quantiles on a Spark DataFrame (Reference §1.5).

Exact quantiles of a huge column require a full sort, so Spark exposes
``DataFrame.approxQuantile(col, probs, relativeError)`` -- a Greenwald-Khanna
sketch. ``relativeError = 0`` gives exact (expensive) results;
``relativeError = 0.01`` is a good default for big data. There is also the SQL
``percentile_approx`` aggregation, handy inside ``groupBy``.

Run:  python quantiles.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)


def quantiles(df, col: str, probs=(0.25, 0.5, 0.75), relative_error: float = 0.01):
    vals = df.approxQuantile(col, list(probs), relative_error)
    return dict(zip(probs, vals))


def five_number_summary(df, col: str, relative_error: float = 0.01):
    mn, mx = df.agg(F.min(col), F.max(col)).first()
    q1, med, q3 = df.approxQuantile(col, [0.25, 0.5, 0.75], relative_error)
    return {"min": mn, "Q1": q1, "median": med, "Q3": q3, "max": mx}


def grouped_median(df, group_col: str, value_col: str):
    """Approximate median of ``value_col`` within each level of ``group_col``."""
    return df.groupBy(group_col).agg(
        F.expr(f"percentile_approx({value_col}, 0.5)").alias("median"),
        F.count(value_col).alias("n"),
    )


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("quantiles").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.createDataFrame([(v,) for v in [3, 7, 8, 5, 12, 14, 21, 13, 18]], ["x"])
        print("quantiles 0.25/0.5/0.75:", quantiles(df, "x", relative_error=0.0))
        print("five-number summary    :", five_number_summary(df, "x", relative_error=0.0))

        g = spark.createDataFrame(
            [("a", 1), ("a", 3), ("a", 5), ("b", 10), ("b", 20), ("b", 30)], ["grp", "val"]
        )
        grouped_median(g, "grp", "val").show()
    finally:
        spark.stop()
