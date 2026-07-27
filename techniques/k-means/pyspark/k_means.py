"""k-means on a Spark DataFrame (Reference §9.9).

Spark MLlib's ``pyspark.ml.clustering.KMeans`` implements k-means|| (parallel
k-means++, Bahmani et al. 2012) -- a distributed init strategy that scales to
huge n. Then the standard Lloyd iterations run distributed.

Run:  python k_means.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.ml.feature import VectorAssembler    # combine columns into a features vector
from pyspark.ml.clustering import KMeans    # MLlib k-means (k-means|| init + Lloyd)
from pyspark.ml.evaluation import ClusteringEvaluator    # silhouette on Spark


def fit_kmeans_spark(df, feature_cols, k: int, seed: int = 0) -> dict:
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features")
    dft = va.transform(df).select("features")
    km = KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster")
    model = km.fit(dft)
    preds = model.transform(dft)
    silh = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                                metricName="silhouette").evaluate(preds)
    return {"centroids": [c.tolist() for c in model.clusterCenters()],
            "inertia": float(model.summary.trainingCost),   # MLlib calls it 'trainingCost'
            "silhouette": float(silh),
            "cluster_sizes": [row["count"] for row in
                              preds.groupBy("cluster").count().orderBy("cluster").collect()],
            "k": k}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("kmeans").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(51)
        rows = []
        for cx, cy in [(0, 0), (4, 4), (8, 0)]:
            for _ in range(200):
                rows.append((random.gauss(cx, 0.7), random.gauss(cy, 0.7)))
        df = spark.createDataFrame(rows, ["x", "y"])
        for k, v in fit_kmeans_spark(df, ["x", "y"], k=3).items():
            print(f"  {k:16s}: {v}")
    finally:
        spark.stop()
