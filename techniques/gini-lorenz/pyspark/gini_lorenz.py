"""Gini coefficient and Lorenz curve on a Spark DataFrame (Reference §1.23).

The Gini coefficient needs the data sorted with a rank attached, which maps onto
a window function:  G = 2*sum(rank_i * x_i) / (n * sum x) - (n + 1)/n  (sorted
ascending, rank = 1..n). The Lorenz curve is the cumulative value share over the
ordered rows. Both are computed without collecting the data to the driver
(except a few scalars at the end).

Run:  python gini_lorenz.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession, Window    # SparkSession: entry point;  Window: window-function specifications
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)


def gini(df, col: str, bias_corrected: bool = False) -> float:
    w = Window.orderBy(col)
    ranked = df.select(F.col(col).cast("double").alias("v")).withColumn("rank", F.row_number().over(w))
    agg = ranked.agg(
        F.sum(F.col("rank") * F.col("v")).alias("weighted_sum"),
        F.sum("v").alias("total"),
        F.count(F.lit(1)).alias("n"),
    ).first()
    n, total, ws = agg["n"], agg["total"], agg["weighted_sum"]
    g = (2.0 * ws) / (n * total) - (n + 1.0) / n
    return g * n / (n - 1.0) if bias_corrected else g


def lorenz_curve(df, col: str):
    """Return a DataFrame [rank, p, L]: cumulative population share and value share."""
    w_order = Window.orderBy(col)
    w_cum = w_order.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    ranked = df.select(F.col(col).cast("double").alias("v")).withColumn("rank", F.row_number().over(w_order))
    n = ranked.count()
    total = ranked.agg(F.sum("v")).first()[0]
    return (ranked
            .withColumn("cum_v", F.sum("v").over(w_cum))
            .select(
                "rank",
                (F.col("rank") / F.lit(n)).alias("p"),
                (F.col("cum_v") / F.lit(total)).alias("L"),
            )
            .orderBy("rank"))


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("gini-lorenz").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.createDataFrame(
            [(v,) for v in [10, 12, 15, 18, 20, 25, 30, 40, 60, 220]], ["income"]
        )
        print("Gini                :", round(gini(df, "income"), 4))
        print("Gini (bias-corrected):", round(gini(df, "income", bias_corrected=True), 4))
        print("Lorenz curve:")
        lorenz_curve(df, "income").show()
    finally:
        spark.stop()
