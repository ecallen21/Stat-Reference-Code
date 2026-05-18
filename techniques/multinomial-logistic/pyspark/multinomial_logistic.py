"""Multinomial logistic regression on a Spark DataFrame (Reference §7.4).

Spark MLlib's ``LogisticRegression(family="multinomial")`` handles K >= 2
outcomes via softmax MLE with L-BFGS.

Run:  python multinomial_logistic.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)
from pyspark.ml.feature import VectorAssembler    # combine multiple columns into a single 'features' vector for MLlib
from pyspark.ml.classification import LogisticRegression    # MLlib logistic regression (binary + multinomial)


def fit_multinomial(df, feature_cols, label_col: str) -> dict:
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features")
    dft = va.transform(df).select("features", F.col(label_col).alias("label"))
    lr = LogisticRegression(featuresCol="features", labelCol="label",
                            regParam=0.0, fitIntercept=True, family="multinomial")
    model = lr.fit(dft)
    s = model.summary
    coef_matrix = model.coefficientMatrix.toArray().tolist()
    intercept = model.interceptVector.toArray().tolist()
    return {"K": model.numClasses,
            "intercept_per_class": intercept,
            "coefficient_matrix": coef_matrix,
            "feature_cols": list(feature_cols),
            "accuracy_train": float(s.accuracy)}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("multinomial").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import math, random
        random.seed(2)
        B = [[0.5, 1.0, -0.5], [-0.3, -0.6, 0.8], [0.0, 0.0, 0.0]]
        rows = []
        for _ in range(2000):
            x1 = random.gauss(0, 1); x2 = random.gauss(0, 1)
            features = [1.0, x1, x2]
            etas = [sum(b * f for b, f in zip(row, features)) for row in B]
            mx = max(etas); exps = [math.exp(e - mx) for e in etas]
            total = sum(exps); p = [e / total for e in exps]
            # sample class 0..2
            r = random.random(); cum = 0; cls = 2
            for i, pi in enumerate(p):
                cum += pi
                if r < cum: cls = i; break
            rows.append((x1, x2, cls))
        df = spark.createDataFrame(rows, ["x1", "x2", "y"])
        for k, v in fit_multinomial(df, ["x1", "x2"], "y").items():
            print(f"  {k:24s}: {v}")
    finally:
        spark.stop()
