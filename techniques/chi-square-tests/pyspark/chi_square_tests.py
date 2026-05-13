"""Chi-square test of independence on a Spark DataFrame (Reference §3.5).

For huge categorical datasets, the distributed part is the contingency table:
groupBy(row_col, col_col).count() -> aggregate the joint counts on the cluster,
collect the small r x c table to the driver, then run the closed-form chi-square
there. Spark MLlib also exposes ``ChiSquareTest`` for the same purpose; this
file shows the explicit pattern so the math is visible.

Run:  python chi_square_tests.py
"""
from __future__ import annotations

import math

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy import stats


def independence_spark(df, row_col: str, col_col: str, correction: bool = False) -> dict:
    """Chi-square test of independence built from a Spark joint-count aggregation."""
    counts = df.groupBy(row_col, col_col).agg(F.count(F.lit(1)).alias("n")).collect()
    row_labels = sorted({r[row_col] for r in counts})
    col_labels = sorted({r[col_col] for r in counts})
    r = len(row_labels); c = len(col_labels)
    ridx = {v: i for i, v in enumerate(row_labels)}
    cidx = {v: j for j, v in enumerate(col_labels)}
    table = [[0] * c for _ in range(r)]
    for row in counts:
        table[ridx[row[row_col]]][cidx[row[col_col]]] = row["n"]

    row_tot = [sum(row) for row in table]
    col_tot = [sum(table[i][j] for i in range(r)) for j in range(c)]
    n = sum(row_tot)
    expected = [[row_tot[i] * col_tot[j] / n for j in range(c)] for i in range(r)]
    if correction and (r, c) == (2, 2):
        chi2 = sum(max(0.0, abs(table[i][j] - expected[i][j]) - 0.5) ** 2
                   / expected[i][j] for i in range(r) for j in range(c))
    else:
        chi2 = sum((table[i][j] - expected[i][j]) ** 2 / expected[i][j]
                   for i in range(r) for j in range(c))
    df_chi = (r - 1) * (c - 1)
    p = float(stats.chi2.sf(chi2, df_chi))
    cramers_v = math.sqrt(chi2 / (n * min(r - 1, c - 1)))
    return {"row_labels": row_labels, "col_labels": col_labels,
            "table": table, "expected": expected,
            "chi_square": chi2, "df": df_chi, "p_value": p,
            "cramers_v": cramers_v, "n": n}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("chi-square").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        rows = ([("North", "Yes")] * 30 + [("North", "No")] * 10 +
                [("South", "Yes")] * 25 + [("South", "No")] * 15 +
                [("East", "Yes")] * 10 + [("East", "No")] * 20)
        df = spark.createDataFrame(rows, ["region", "outcome"])
        res = independence_spark(df, "region", "outcome")
        print(f"chi^2 = {res['chi_square']:.4f}  df = {res['df']}  p = {res['p_value']:.6f}")
        print(f"Cramer's V = {res['cramers_v']:.4f}")
        print(f"row labels = {res['row_labels']}")
        print(f"col labels = {res['col_labels']}")
        print(f"observed   = {res['table']}")
        print(f"expected   = {[[round(v, 2) for v in r] for r in res['expected']]}")
    finally:
        spark.stop()
