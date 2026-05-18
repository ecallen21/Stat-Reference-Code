"""Kernel Density Estimation on a Spark DataFrame (Reference §6.21).

Spark MLlib ships ``pyspark.mllib.stat.KernelDensity`` (Gaussian kernel only).
For other kernels, a per-row aggregation works fine for moderate-sized samples
(the trick is that f_hat(grid_j) = sum_i K((grid_j - x_i)/h) / (n h) -- one
linear pass per grid point).

For very large n, sample to a manageable size first, then run the Python KDE on
the sample.

Run:  python kernel_density_estimation.py
"""
from __future__ import annotations

import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def kde_mllib(df, col: str, grid, bandwidth: float | None = None):
    """Gaussian KDE via MLlib's KernelDensity (Spark 2+).

    ``grid`` is a list of x-values where we want to evaluate f_hat.
    """
    from pyspark.mllib.stat import KernelDensity
    rdd = df.select(col).rdd.map(lambda r: float(r[0]))
    if bandwidth is None:
        # Silverman: 0.9 * min(sd, IQR/1.34) * n^(-1/5)
        stats_row = df.agg(F.stddev_samp(col).alias("sd"),
                           F.expr(f"percentile_approx({col}, 0.25)").alias("q1"),
                           F.expr(f"percentile_approx({col}, 0.75)").alias("q3"),
                           F.count(F.lit(1)).alias("n")).first()
        h = 0.9 * min(stats_row["sd"], (stats_row["q3"] - stats_row["q1"]) / 1.34) \
            * stats_row["n"] ** (-0.2)
    else:
        h = bandwidth
    kd = KernelDensity()
    kd.setSample(rdd); kd.setBandwidth(h)
    return list(zip(grid, kd.estimate(grid))), h


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("kde").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(8)
        rows = [(random.gauss(-2, 0.8),) for _ in range(500)] + \
               [(random.gauss(2, 1.0),) for _ in range(750)]
        df = spark.createDataFrame(rows, ["x"])

        grid = [-4, -2, 0, 2, 4]
        pts, h = kde_mllib(df, "x", grid)
        print(f"Silverman h = {h:.4f}")
        for g, density in pts:
            print(f"  f_hat({g:+.1f}) = {density:.4f}")
    finally:
        spark.stop()
