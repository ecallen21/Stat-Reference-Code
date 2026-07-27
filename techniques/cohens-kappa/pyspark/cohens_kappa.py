"""Cohen's kappa on a Spark DataFrame (Reference §8.4).

The distributed part is building the K x K confusion matrix from potentially
billions of rated items. ``groupBy(rater1, rater2).count()`` aggregates to at
most K^2 rows, which we collect and finish on the driver.

Run:  python cohens_kappa.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def cohens_kappa_spark(df, rater1_col: str, rater2_col: str) -> dict:
    """Cohen's kappa on a Spark DataFrame with two rating columns."""
    agg = (df.groupBy(rater1_col, rater2_col)
             .agg(F.count(F.lit(1)).alias("n"))
             .collect())
    cats = sorted({r[rater1_col] for r in agg} | {r[rater2_col] for r in agg})
    idx = {c: i for i, c in enumerate(cats)}
    K = len(cats)
    m = np.zeros((K, K), dtype=float)
    for r in agg:
        m[idx[r[rater1_col]], idx[r[rater2_col]]] = r["n"]
    n = m.sum()
    p_o = np.trace(m) / n
    row = m.sum(axis=1) / n
    col = m.sum(axis=0) / n
    p_e = float((row * col).sum())
    if p_e == 1.0:
        kappa = 1.0 if p_o == 1.0 else 0.0
        se = float("nan")
    else:
        kappa = (p_o - p_e) / (1 - p_e)
        # Fleiss ASE
        diag_p = np.diag(m) / n
        A = ((diag_p / (1 - p_e)) * (1 - (row + col) * (1 - kappa))).sum()
        off = 0.0
        for i in range(K):
            for j in range(K):
                if i == j: continue
                off += m[i, j] / n * (col[i] + row[j]) ** 2
        B = ((1 - kappa) ** 2) / (1 - p_e) ** 2 * off
        C = (kappa - p_e * (1 - kappa)) ** 2 / (1 - p_e) ** 2
        se = math.sqrt(max((A + B - C) / n, 0.0))
    return {"categories": cats,
            "kappa": float(kappa),
            "ASE": float(se),
            "p_observed": float(p_o),
            "p_expected": p_e,
            "n": int(n),
            "K": K}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("kappa").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(3)
        cats = ["low", "medium", "high"]
        rows = []
        for _ in range(2000):
            r1 = random.choice(cats)
            r2 = r1 if random.random() < 0.7 else random.choice([c for c in cats if c != r1])
            rows.append((r1, r2))
        df = spark.createDataFrame(rows, ["rater1", "rater2"])
        for k, v in cohens_kappa_spark(df, "rater1", "rater2").items():
            print(f"  {k:12s}: {v}")
    finally:
        spark.stop()
