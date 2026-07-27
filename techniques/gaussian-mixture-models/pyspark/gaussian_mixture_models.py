"""Gaussian Mixture Models on a Spark DataFrame (Reference §9.12).

Spark MLlib's ``pyspark.ml.clustering.GaussianMixture`` runs EM on distributed
data. Convergence monitoring, weights, means, and covariance matrices are all
retrievable from the fitted model.

Run:  python gaussian_mixture_models.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.ml.feature import VectorAssembler    # combine columns into a features vector
from pyspark.ml.clustering import GaussianMixture    # MLlib GMM via EM


def fit_gmm_spark(df, feature_cols, K: int, seed: int = 0) -> dict:
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features")
    dft = va.transform(df).select("features")
    gm = GaussianMixture(k=K, featuresCol="features", predictionCol="cluster",
                         probabilityCol="posteriors", seed=seed, maxIter=200)
    model = gm.fit(dft)
    return {"weights": list(model.weights),
            "means": [row["mean"].toArray().tolist() for row in model.gaussiansDF.collect()],
            "log_lik": float(model.summary.logLikelihood),
            "cluster_sizes": [row["count"] for row in
                              model.summary.cluster.groupBy("cluster")
                              .count().orderBy("cluster").collect()],
            "K": K}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("gmm").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(71)
        rows = []
        for cx, cy in [(0, 0), (5, 5), (1, 3)]:
            for _ in range(300):
                rows.append((random.gauss(cx, 1.0), random.gauss(cy, 1.0)))
        df = spark.createDataFrame(rows, ["x", "y"])
        for k, v in fit_gmm_spark(df, ["x", "y"], K=3).items():
            print(f"  {k:14s}: {v}")
    finally:
        spark.stop()
