"""Coefficient of variation on a Spark DataFrame (Reference §1.22 and §1.33).

CV = stddev / mean is a one-pass aggregation. The geometric CV needs the SD of
log(x). Within-subject CV (assay precision over many subjects) is a groupBy
variance followed by a global average -- a natural distributed computation.

Run:  python coefficient_of_variation.py
"""
from __future__ import annotations

import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def cv(df, col: str, as_percent: bool = False):
    row = df.agg(F.stddev_samp(col).alias("sd"), F.mean(col).alias("mean")).first()
    if row["mean"] == 0:
        raise ValueError("CV undefined: mean is zero")
    v = row["sd"] / row["mean"]
    return v * 100 if as_percent else v


def geometric_cv(df, col: str, as_percent: bool = False):
    s_log = df.agg(F.stddev_samp(F.log(col)).alias("s")).first()["s"]
    v = math.sqrt(math.exp(s_log ** 2) - 1.0)
    return v * 100 if as_percent else v


def within_subject_cv(df, subject_col: str, value_col: str, as_percent: bool = False):
    """sqrt(mean of per-subject variances) / overall mean -- assay precision style."""
    per_subject = df.groupBy(subject_col).agg(F.var_samp(value_col).alias("v"))
    mean_var = per_subject.agg(F.mean("v")).first()[0]
    overall_mean = df.agg(F.mean(value_col)).first()[0]
    v = math.sqrt(mean_var) / overall_mean
    return v * 100 if as_percent else v


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("coefficient-of-variation").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.createDataFrame(
            [(v,) for v in [98.2, 101.4, 99.7, 100.1, 102.3, 97.9, 100.8]], ["x"]
        )
        print("CV          :", round(cv(df, "x", as_percent=True), 3), "%")
        print("geometric CV:", round(geometric_cv(df, "x", as_percent=True), 3), "%")

        reps = spark.createDataFrame(
            [(1, 10.1), (1, 10.3), (1, 9.8), (2, 12.0), (2, 11.7), (2, 12.2),
             (3, 9.5), (3, 9.7), (3, 9.4), (4, 11.2), (4, 11.0), (4, 11.5)],
            ["subject", "value"],
        )
        print("within-subject CV:", round(within_subject_cv(reps, "subject", "value", True), 3), "%")
    finally:
        spark.stop()
