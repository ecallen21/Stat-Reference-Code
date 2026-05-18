"""Simple linear regression on a Spark DataFrame (Reference §5.1).

Two routes shown:
  1. Closed form via sufficient statistics (one pass; what we'd do by hand).
  2. ``pyspark.ml.regression.LinearRegression`` (Spark MLlib).

The sufficient-stat route mirrors the from-scratch Python version exactly --
useful when you want diagnostics that MLlib doesn't expose, and avoids the
cost of building a fitted MLlib model when all you want is the slope.

Run:  python simple_linear_regression.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def fit_spark(df, x_col: str, y_col: str) -> dict:
    """Closed-form OLS via sufficient statistics (single ``agg``)."""
    row = df.agg(
        F.count(F.lit(1)).alias("n"),
        F.mean(x_col).alias("mx"), F.mean(y_col).alias("my"),
        F.sum(F.col(x_col) * F.col(x_col)).alias("xx"),
        F.sum(F.col(x_col) * F.col(y_col)).alias("xy"),
        F.sum(F.col(y_col) * F.col(y_col)).alias("yy"),
    ).first().asDict()
    n, mx, my = row["n"], row["mx"], row["my"]
    Sxx = row["xx"] - n * mx * mx
    Sxy = row["xy"] - n * mx * my
    Syy = row["yy"] - n * my * my
    b1 = Sxy / Sxx
    b0 = my - b1 * mx
    rss = Syy - b1 * Sxy
    df_r = n - 2
    sigma = math.sqrt(rss / df_r)
    se_b1 = sigma / math.sqrt(Sxx)
    se_b0 = sigma * math.sqrt(1 / n + mx * mx / Sxx)
    t_b1 = b1 / se_b1
    return {"n": n, "beta_0": b0, "beta_1": b1,
            "se_beta_0": se_b0, "se_beta_1": se_b1,
            "t_beta_1": t_b1,
            "p_beta_1": float(2 * stats.t.sf(abs(t_b1), df_r)),
            "df_residual": df_r, "sigma_hat": sigma,
            "r_squared": 1 - rss / Syy}


def fit_mllib(df, x_col: str, y_col: str):
    """The MLlib route -- builds a VectorAssembler feature and fits LinearRegression."""
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression
    va = VectorAssembler(inputCols=[x_col], outputCol="features")
    dft = va.transform(df).select("features", F.col(y_col).alias("label"))
    lr = LinearRegression(featuresCol="features", labelCol="label",
                          regParam=0.0, fitIntercept=True)
    model = lr.fit(dft)
    s = model.summary
    return {"beta_0": model.intercept, "beta_1": float(model.coefficients[0]),
            "r_squared": s.r2, "rmse": s.rootMeanSquaredError}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("simple-lr").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(0)
        rows = [(random.uniform(0, 10),) for _ in range(200)]
        rows = [(x, 2.0 + 0.8 * x + random.gauss(0, 1.5)) for (x,) in rows]
        df = spark.createDataFrame(rows, ["x", "y"])
        print("Closed form (sufficient stats):")
        for k, v in fit_spark(df, "x", "y").items():
            print(f"  {k:11s}: {v}")
        print("\nMLlib LinearRegression:")
        for k, v in fit_mllib(df, "x", "y").items():
            print(f"  {k:11s}: {v}")
    finally:
        spark.stop()
