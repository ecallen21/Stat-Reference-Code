"""Nonparametric bootstrap on a Spark DataFrame (Reference §10.1).

The distributed part: with n and B large, running B bootstrap replicates
where each resamples n rows serially on the driver is prohibitive. Spark
lets us:
    - Broadcast the (small) statistic function.
    - For each of B replicates, do ``df.sample(withReplacement=True, fraction=1.0)``
      to draw a resample; compute the statistic on the resample.

Alternatively, on very large data, the "bag of little bootstraps" (Kleiner et
al., 2014) subsamples b << n rows per replicate to trade some accuracy for
massive speed. This file shows the straightforward full-resample version.

Run:  python nonparametric_bootstrap.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)


def bootstrap_mean_spark(df, value_col: str, n_boot: int = 200,
                          conf: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap the MEAN of a Spark DataFrame column with case resampling."""
    n = df.count()
    theta_hat = float(df.agg(F.mean(F.col(value_col))).collect()[0][0])
    theta_star = np.empty(n_boot)
    for b in range(n_boot):
        boot = df.sample(withReplacement=True, fraction=1.0, seed=seed + b)
        theta_star[b] = float(boot.agg(F.mean(F.col(value_col))).collect()[0][0])
    alpha = 1 - conf
    lo, hi = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])
    return {"theta_hat": theta_hat,
            "bootstrap_SE": float(theta_star.std(ddof=1)),
            "CI_percentile": {"lower": float(lo), "upper": float(hi)},
            "n_boot": n_boot, "n": n, "conf": conf,
            "method": "Spark bootstrap (mean, full-resample, case resampling)"}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("boot").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(4)
        rows = [(random.expovariate(1 / 2.0),) for _ in range(2000)]
        df = spark.createDataFrame(rows, ["y"])
        for k, v in bootstrap_mean_spark(df, "y", n_boot=200).items():
            print(f"  {k:16s}: {v}")
    finally:
        spark.stop()
