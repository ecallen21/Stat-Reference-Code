"""Poisson regression on a Spark DataFrame (Reference §7.12, §7.43).

Spark MLlib's ``GeneralizedLinearRegression(family="poisson", link="log")``
fits Poisson GLM by IRLS. Offset is supported via ``offsetCol``.

Run:  python poisson_regression.py
"""
from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression


def fit_poisson(df, feature_cols, label_col: str,
                offset_col: str | None = None) -> dict:
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features")
    dft = va.transform(df).select("features", F.col(label_col).alias("label"),
                                  *( [F.col(offset_col).alias("offset")] if offset_col else [] ))
    kwargs = {"family": "poisson", "link": "log",
              "featuresCol": "features", "labelCol": "label",
              "fitIntercept": True, "regParam": 0.0}
    if offset_col is not None:
        kwargs["offsetCol"] = "offset"
    glr = GeneralizedLinearRegression(**kwargs)
    model = glr.fit(dft)
    s = model.summary
    return {"intercept": float(model.intercept),
            "coefficients": dict(zip(feature_cols,
                                     [float(c) for c in model.coefficients])),
            "deviance": float(s.deviance),
            "dispersion": float(s.dispersion),
            "aic": float(s.aic())}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("poisson").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import math, random
        random.seed(4)
        rows = []
        for _ in range(1000):
            x1 = random.gauss(0, 1); x2 = random.gauss(0, 1)
            exposure = random.uniform(0.5, 3.0)
            mu = exposure * math.exp(0.5 + 0.6 * x1 - 0.3 * x2)
            # Sample Poisson via inverse CDF approximation -- use random's gauss + round
            # Simpler: just use the mean as Poisson approx for demo size
            import math as m
            # Quick Poisson sampler
            L = m.exp(-mu); k = 0; p = 1.0
            while p > L:
                k += 1; p *= random.random()
            y = k - 1
            rows.append((x1, x2, m.log(exposure), float(y)))
        df = spark.createDataFrame(rows, ["x1", "x2", "log_exposure", "y"])
        for k, v in fit_poisson(df, ["x1", "x2"], "y", offset_col="log_exposure").items():
            print(f"  {k:14s}: {v}")
    finally:
        spark.stop()
