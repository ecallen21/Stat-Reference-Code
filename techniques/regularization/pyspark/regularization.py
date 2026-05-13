"""Regularized linear regression on a Spark DataFrame (Reference §5.9, §5.17).

Spark MLlib's ``LinearRegression`` exposes both penalties via two parameters:

  regParam     = lambda
  elasticNetParam = alpha in [0, 1]
        alpha = 0  -> pure L2 (ridge)
        alpha = 1  -> pure L1 (lasso)
        in between -> elastic net

For CV, ``pyspark.ml.tuning.CrossValidator`` + ``ParamGridBuilder`` is the
standard idiom. We show all three fits below.

Run:  python regularization.py
"""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


def fit_regularized(df, feature_cols, label_col,
                    reg_param: float, l1_ratio: float = 0.0):
    """Fit Spark MLlib LinearRegression with elastic-net penalty."""
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features")
    dft = va.transform(df).select("features", F.col(label_col).alias("label"))
    lr = LinearRegression(featuresCol="features", labelCol="label",
                          regParam=reg_param, elasticNetParam=l1_ratio,
                          fitIntercept=True, standardization=True)
    model = lr.fit(dft)
    return {"intercept": model.intercept,
            "coefficients": dict(zip(feature_cols,
                                     [float(c) for c in model.coefficients])),
            "r_squared": model.summary.r2,
            "rmse": model.summary.rootMeanSquaredError,
            "regParam": reg_param, "elasticNetParam": l1_ratio,
            "kind": ("ridge" if l1_ratio == 0
                     else "lasso" if l1_ratio == 1
                     else "elastic_net")}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("regularization").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random; random.seed(4)
        true_beta = [2.0, -1.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.3]
        rows = []
        for _ in range(200):
            x = [random.gauss(0, 1) for _ in range(10)]
            y = sum(b * xi for b, xi in zip(true_beta, x)) + random.gauss(0, 1)
            rows.append(tuple(x + [y]))
        cols = [f"x{i+1}" for i in range(10)] + ["y"]
        df = spark.createDataFrame(rows, cols)

        for kind, l1 in [("ridge", 0.0), ("lasso", 1.0), ("elastic_net", 0.5)]:
            print(f"\n=== {kind} (regParam = 0.1, l1_ratio = {l1}) ===")
            res = fit_regularized(df, cols[:-1], "y", reg_param=0.1, l1_ratio=l1)
            print(f"  intercept = {res['intercept']:.4f}")
            for nm, b in res["coefficients"].items():
                print(f"  {nm}: {b:+.4f}")
            print(f"  R^2 = {res['r_squared']:.4f}   RMSE = {res['rmse']:.4f}")
    finally:
        spark.stop()
