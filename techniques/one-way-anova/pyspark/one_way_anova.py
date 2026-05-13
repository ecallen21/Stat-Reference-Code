"""One-way ANOVA on a Spark DataFrame (Reference §3.8).

For huge data, the ANOVA reduces to per-group sufficient statistics: n_i,
mean_i, var_i. A single ``groupBy`` aggregation gives all three; the closed-form
F-test (classic and Welch) is computed on the driver.

Run:  python one_way_anova.py
"""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy import stats


def _collect_stats(df, value_col, group_col):
    return (df.groupBy(group_col)
              .agg(F.count(value_col).alias("n"),
                   F.mean(value_col).alias("mean"),
                   F.var_samp(value_col).alias("var"))
              .orderBy(group_col)
              .collect())


def classic_anova_spark(df, value_col: str, group_col: str) -> dict:
    rows = _collect_stats(df, value_col, group_col)
    ns = [r["n"] for r in rows]; means = [r["mean"] for r in rows]
    vars_ = [r["var"] for r in rows]
    N = sum(ns); k = len(ns)
    grand = sum(n * m for n, m in zip(ns, means)) / N
    ss_b = sum(n * (m - grand) ** 2 for n, m in zip(ns, means))
    ss_w = sum((n - 1) * v for n, v in zip(ns, vars_))
    df1, df2 = k - 1, N - k
    F_stat = (ss_b / df1) / (ss_w / df2)
    p = float(stats.f.sf(F_stat, df1, df2))
    return {"groups": [r[group_col] for r in rows],
            "n_per_group": ns, "means": means,
            "F": F_stat, "df1": df1, "df2": df2, "p_value": p}


def welch_anova_spark(df, value_col: str, group_col: str) -> dict:
    rows = _collect_stats(df, value_col, group_col)
    ns = [r["n"] for r in rows]; means = [r["mean"] for r in rows]
    vars_ = [r["var"] for r in rows]
    k = len(ns)
    w = [n / v for n, v in zip(ns, vars_)]
    W = sum(w)
    grand = sum(wi * m for wi, m in zip(w, means)) / W
    num = sum(wi * (m - grand) ** 2 for wi, m in zip(w, means)) / (k - 1)
    denom = 1 + (2 * (k - 2) / (k * k - 1)) * sum(
        (1 - wi / W) ** 2 / (n - 1) for wi, n in zip(w, ns)
    )
    F_stat = num / denom
    df1 = k - 1
    df2 = (k * k - 1) / (3 * sum((1 - wi / W) ** 2 / (n - 1) for wi, n in zip(w, ns)))
    p = float(stats.f.sf(F_stat, df1, df2))
    return {"groups": [r[group_col] for r in rows],
            "F": F_stat, "df1": df1, "df2": df2, "p_value": p, "method": "Welch"}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("one-way-anova").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(2)
        rows = ([("A", random.gauss(50, 10)) for _ in range(30)] +
                [("B", random.gauss(55, 10)) for _ in range(28)] +
                [("C", random.gauss(60, 18)) for _ in range(32)])
        df = spark.createDataFrame(rows, ["arm", "score"])
        print("Classic ANOVA:")
        for k, v in classic_anova_spark(df, "score", "arm").items():
            print(f"  {k:14s}: {v}")
        print("\nWelch ANOVA:")
        for k, v in welch_anova_spark(df, "score", "arm").items():
            print(f"  {k:14s}: {v}")
    finally:
        spark.stop()
