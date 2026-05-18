"""Empirical CDF on a Spark DataFrame (Reference §1.13).

Compute the ECDF as a table: for each distinct value v, the cumulative count of
rows with x <= v, divided by n. This is a group-by + running sum over an ordered
window -- a standard distributed pattern. Evaluating F_n at arbitrary points is
then a join/lookup against that table.

Run:  python ecdf.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession, Window    # SparkSession: entry point;  Window: window-function specifications
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)


def ecdf_table(df, col: str):
    """Return a DataFrame [value, count, cum_count, Fn] giving the ECDF."""
    n = df.count()
    counts = df.groupBy(F.col(col).alias("value")).agg(F.count(F.lit(1)).alias("count"))
    w = Window.orderBy("value").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return (counts
            .withColumn("cum_count", F.sum("count").over(w))
            .withColumn("Fn", F.col("cum_count") / F.lit(n))
            .orderBy("value"))


def evaluate(ecdf_tbl, points):
    """F_n(t) for each t in ``points``: largest Fn whose value <= t (0 if none)."""
    rows = [(float(p),) for p in points]
    pts = ecdf_tbl.sparkSession.createDataFrame(rows, ["t"])
    joined = pts.join(ecdf_tbl, ecdf_tbl["value"] <= pts["t"], "left")
    return (joined.groupBy("t")
            .agg(F.coalesce(F.max("Fn"), F.lit(0.0)).alias("Fn"))
            .orderBy("t"))


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("ecdf").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.createDataFrame([(v,) for v in [2, 3, 3, 5, 8, 13, 21]], ["x"])
        tbl = ecdf_table(df, "x")
        tbl.show()
        evaluate(tbl, [1, 3, 4, 8, 25]).show()
    finally:
        spark.stop()
