"""Multiple linear regression on a Spark DataFrame (Reference §5.2).

Use ``pyspark.ml.regression.LinearRegression`` for the fit; pull the summary
diagnostics from ``model.summary``. For the from-scratch / sufficient-stats
form, the trick generalizes nicely: a single ``agg`` over X'X and X'y gives
the normal equations, which we solve on the driver.

Run:  python multiple_linear_regression.py
"""
from __future__ import annotations

import math

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from scipy import stats


def fit_sufficient_stats(df, feature_cols, label_col):
    """OLS via X'X and X'y aggregated in one Spark pass.

    Adds an intercept column of 1s. Returns coefficient estimates + SEs from
    the normal equations solved on the driver.
    """
    p = len(feature_cols) + 1
    cols = ["__one"] + list(feature_cols)
    df2 = df.withColumn("__one", F.lit(1.0))

    # X'X (p x p) and X'y (p) and y'y and n -- all in one agg
    agg_exprs = []
    for i in range(p):
        for j in range(i, p):
            agg_exprs.append(F.sum(F.col(cols[i]) * F.col(cols[j])).alias(f"xx_{i}_{j}"))
    for i in range(p):
        agg_exprs.append(F.sum(F.col(cols[i]) * F.col(label_col)).alias(f"xy_{i}"))
    agg_exprs += [F.sum(F.col(label_col) * F.col(label_col)).alias("yy"),
                  F.count(F.lit(1)).alias("n"),
                  F.mean(label_col).alias("ybar")]
    row = df2.agg(*agg_exprs).first().asDict()

    XtX = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            XtX[i, j] = row[f"xx_{i}_{j}"]; XtX[j, i] = XtX[i, j]
    Xty = np.array([row[f"xy_{i}"] for i in range(p)])
    n = row["n"]; yy = row["yy"]; ybar = row["ybar"]

    beta = np.linalg.solve(XtX, Xty)
    rss = yy - beta @ Xty
    df_r = n - p
    sigma2 = rss / df_r; sigma = math.sqrt(sigma2)
    var_beta = sigma2 * np.linalg.pinv(XtX)
    se_beta = np.sqrt(np.diag(var_beta))
    tss = yy - n * ybar * ybar
    return {"n": n, "p": p, "names": cols,
            "beta": beta.tolist(), "se_beta": se_beta.tolist(),
            "rss": rss, "sigma_hat": sigma, "df_residual": df_r,
            "r_squared": 1 - rss / tss}


def fit_mllib(df, feature_cols, label_col, reg_param: float = 0.0):
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features")
    dft = va.transform(df).select("features", F.col(label_col).alias("label"))
    lr = LinearRegression(featuresCol="features", labelCol="label",
                          regParam=reg_param, fitIntercept=True)
    model = lr.fit(dft)
    s = model.summary
    return {"intercept": model.intercept,
            "coefficients": dict(zip(feature_cols, [float(c) for c in model.coefficients])),
            "r_squared": s.r2, "adj_r_squared": s.r2adj,
            "rmse": s.rootMeanSquaredError, "n": s.numInstances}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("multi-lr").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(0)
        rows = []
        for _ in range(200):
            x1 = random.gauss(0, 1); x2 = random.gauss(0, 1); x3 = random.gauss(0, 1)
            y = 1 + 2 * x1 - 1.5 * x2 + 0 * x3 + random.gauss(0, 1)
            rows.append((x1, x2, x3, y))
        df = spark.createDataFrame(rows, ["x1", "x2", "x3", "y"])

        print("Sufficient stats (closed-form OLS):")
        res = fit_sufficient_stats(df, ["x1", "x2", "x3"], "y")
        for nm, b, se in zip(res["names"], res["beta"], res["se_beta"]):
            print(f"  {nm:8s}: {b:+.4f}  SE = {se:.4f}")
        print(f"  R^2 = {res['r_squared']:.4f}   sigma = {res['sigma_hat']:.4f}")

        print("\nMLlib LinearRegression:")
        for k, v in fit_mllib(df, ["x1", "x2", "x3"], "y").items():
            print(f"  {k}: {v}")
    finally:
        spark.stop()
