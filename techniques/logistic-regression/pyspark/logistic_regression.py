"""Binary logistic regression on a Spark DataFrame (Reference §7.1).

Spark MLlib's ``LogisticRegression`` fits binary logistic by L-BFGS / OWL-QN.
Output mirrors ``GeneralizedLinearRegression(family="binomial")`` but the
former is the conventional choice for pure classification.

Run:  python logistic_regression.py
"""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression


def fit_logistic(df, feature_cols, label_col: str) -> dict:
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features")
    dft = va.transform(df).select("features", F.col(label_col).alias("label"))
    lr = LogisticRegression(featuresCol="features", labelCol="label",
                            regParam=0.0, fitIntercept=True, family="binomial")
    model = lr.fit(dft)
    s = model.summary
    return {"intercept": float(model.intercept),
            "coefficients": dict(zip(feature_cols,
                                     [float(c) for c in model.coefficients])),
            "areaUnderROC": float(s.areaUnderROC),
            "accuracy_train": float(s.accuracy),
            "objective_history_last": s.objectiveHistory[-1]}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("logit").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import math, random
        random.seed(0)
        rows = []
        for _ in range(2000):
            x1 = random.gauss(0, 1); x2 = random.gauss(0, 1)
            eta = -0.5 + 1.0 * x1 - 0.7 * x2
            p = 1 / (1 + math.exp(-eta))
            y = 1 if random.random() < p else 0
            rows.append((x1, x2, y))
        df = spark.createDataFrame(rows, ["x1", "x2", "y"])
        for k, v in fit_logistic(df, ["x1", "x2"], "y").items():
            print(f"  {k:24s}: {v}")
    finally:
        spark.stop()
