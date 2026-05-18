"""Two-sample t-tests on a Spark DataFrame (Reference §3.4).

For a two-sample t-test, the only thing the cluster has to compute is
``n / mean / var`` per group -- a single ``groupBy`` aggregation. We then build
Student's or Welch's t and the Satterthwaite df on the driver from those four
scalars, using scipy's t-distribution for the p-value. This is the standard
"distributed sufficient-statistics" pattern: heavy lifting on the workers, tiny
arithmetic on the driver.

Run:  python t_tests.py
"""
from __future__ import annotations

import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy import stats


def two_sample_t_spark(df, value_col: str, group_col: str, equal_var: bool = False,
                       alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """Two-sample t-test from group sufficient statistics in a Spark DataFrame.

    Expects exactly two distinct values of ``group_col``.
    """
    agg = (df.groupBy(group_col)
             .agg(F.count(value_col).alias("n"),
                  F.mean(value_col).alias("mean"),
                  F.var_samp(value_col).alias("var"))
             .collect())
    if len(agg) != 2:
        raise ValueError(f"need exactly 2 groups, found {len(agg)}")
    a, b = sorted(agg, key=lambda r: r[group_col])  # deterministic order
    n1, m1, v1 = a["n"], a["mean"], a["var"]
    n2, m2, v2 = b["n"], b["mean"], b["var"]
    diff = m1 - m2

    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        df_t = n1 + n2 - 2
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df_t = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
    t = diff / se
    if alternative == "two-sided":
        p = 2 * stats.t.sf(abs(t), df_t)
    elif alternative == "greater":
        p = float(stats.t.sf(t, df_t))
    else:
        p = float(stats.t.cdf(t, df_t))

    tcrit = stats.t.ppf(0.5 + conf / 2, df_t)
    return {
        "group1": a[group_col], "group2": b[group_col],
        "n1": n1, "n2": n2, "mean1": m1, "mean2": m2, "var1": v1, "var2": v2,
        "mean_diff": diff, "se": se, "t": t, "df": df_t, "p_value": p,
        "ci_lower": diff - tcrit * se, "ci_upper": diff + tcrit * se,
        "method": "Student" if equal_var else "Welch",
    }


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("t-tests").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(7)
        rows = [("A", random.gauss(105, 14)) for _ in range(30)] + \
               [("B", random.gauss(100, 18)) for _ in range(27)]
        df = spark.createDataFrame(rows, ["arm", "outcome"])
        print("Welch's t-test by arm:")
        for k, v in two_sample_t_spark(df, "outcome", "arm").items():
            print(f"  {k:11s}: {v}")
    finally:
        spark.stop()
