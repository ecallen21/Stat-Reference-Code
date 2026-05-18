"""Gamma regression on a Spark DataFrame (Reference §7.25).

Spark MLlib's ``GeneralizedLinearRegression(family="gamma", link="log")``
fits Gamma GLM by IRLS.

Run:  python gamma_regression.py
"""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression


def fit_gamma(df, feature_cols, label_col: str) -> dict:
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features")
    dft = va.transform(df).select("features", F.col(label_col).alias("label"))
    glr = GeneralizedLinearRegression(family="gamma", link="log",
                                       featuresCol="features", labelCol="label",
                                       fitIntercept=True, regParam=0.0)
    model = glr.fit(dft); s = model.summary
    return {"intercept": float(model.intercept),
            "coefficients": dict(zip(feature_cols,
                                     [float(c) for c in model.coefficients])),
            "deviance": float(s.deviance),
            "dispersion": float(s.dispersion),
            "aic": float(s.aic())}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("gamma").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import math, random; random.seed(8)
        rows = []
        for _ in range(1000):
            x1 = random.gauss(0, 1); x2 = random.gauss(0, 1)
            mu = math.exp(2 + 0.5 * x1 - 0.3 * x2)
            # Gamma sampling via rejection on standard normal isn't trivial;
            # use Marsaglia-Tsang via random.gammavariate
            y = random.gammavariate(4.0, mu / 4.0)
            rows.append((x1, x2, y))
        df = spark.createDataFrame(rows, ["x1", "x2", "y"])
        for k, v in fit_gamma(df, ["x1", "x2"], "y").items():
            print(f"  {k:14s}: {v}")
    finally:
        spark.stop()
