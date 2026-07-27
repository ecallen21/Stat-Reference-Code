"""PCA on a Spark DataFrame (Reference §9.3).

Spark MLlib's ``pyspark.ml.feature.PCA`` computes principal components on a
distributed feature vector. It returns the top-k principal components (loadings)
and can transform new data. Explained-variance ratios are also available on the
fitted model.

Run:  python pca.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.ml.feature import PCA, VectorAssembler    # PCA transformer + feature-vector assembler
from pyspark.ml.stat import Summarizer                 # column statistics (mean, variance) on DataFrames


def fit_pca_spark(df, feature_cols, k: int) -> dict:
    """Fit MLlib PCA and return loadings + explained variance ratios."""
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features_raw")
    dft = va.transform(df).select("features_raw")
    # Center: subtract mean vector (MLlib PCA does NOT center by default)
    mean_row = dft.select(Summarizer.mean(dft["features_raw"])).first()[0]
    import numpy as np
    mean_vec = np.array(mean_row.toArray())
    # Convert to numpy, center, reassemble as a Spark DataFrame of Vectors
    from pyspark.ml.linalg import Vectors
    centered = dft.rdd.map(lambda r: (Vectors.dense(np.array(r[0].toArray()) - mean_vec),))
    dft2 = centered.toDF(["features"])
    pca = PCA(k=k, inputCol="features", outputCol="pc")
    model = pca.fit(dft2)
    total_var = float(sum(model.explainedVariance.toArray()))
    return {
        "n_components": k,
        "loadings": model.pc.toArray().tolist(),         # p x k matrix
        "explained_variance_ratio": model.explainedVariance.toArray().tolist(),
        "cumulative_variance_ratio": [float(v) for v in
            list(__import__("numpy").cumsum(model.explainedVariance.toArray()))],
        "center": mean_vec.tolist(),
        "feature_cols": list(feature_cols),
    }


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("pca").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(23)
        rows = []
        for _ in range(500):
            s1 = random.gauss(0, 3.0); s2 = random.gauss(0, 1.5)
            x1 = 0.6 * s1 + 0.1 * s2 + random.gauss(0, 0.3) + 1
            x2 = 0.5 * s1 - 0.2 * s2 + random.gauss(0, 0.3) + 2
            x3 = 0.4 * s1 + 0.3 * s2 + random.gauss(0, 0.3) + 3
            x4 = 0.3 * s1 - 0.5 * s2 + random.gauss(0, 0.3) + 4
            x5 = 0.2 * s1 + 0.7 * s2 + random.gauss(0, 0.3) + 5
            rows.append((x1, x2, x3, x4, x5))
        df = spark.createDataFrame(rows, ["x1", "x2", "x3", "x4", "x5"])
        out = fit_pca_spark(df, ["x1", "x2", "x3", "x4", "x5"], k=3)
        print("=== MLlib PCA ===")
        print(f"  explained variance ratio: {out['explained_variance_ratio']}")
        print(f"  cumulative:               {out['cumulative_variance_ratio']}")
    finally:
        spark.stop()
