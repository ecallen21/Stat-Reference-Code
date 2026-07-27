"""Cross-validation on a Spark DataFrame (Reference §10.8).

Spark MLlib's ``pyspark.ml.tuning.CrossValidator`` runs K-fold CV against a
grid of hyperparameters, retraining the pipeline on each fold. Works with any
MLlib Estimator/Pipeline.

Run:  python cross_validation.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession    # Spark entry point (build / get a SparkSession)
from pyspark.ml.feature import VectorAssembler    # combine columns into a features vector
from pyspark.ml.regression import LinearRegression    # MLlib linear regression
from pyspark.ml.evaluation import RegressionEvaluator    # MSE / RMSE evaluators on Spark
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder    # k-fold CV + param grids


def cv_linear_regression(df, feature_cols, label_col: str, k: int = 5,
                          reg_grid=(0.0, 0.01, 0.1)) -> dict:
    va = VectorAssembler(inputCols=list(feature_cols), outputCol="features")
    dft = va.transform(df).select("features", label_col)
    lr = LinearRegression(featuresCol="features", labelCol=label_col,
                          maxIter=50, elasticNetParam=0.0)
    grid = ParamGridBuilder().addGrid(lr.regParam, list(reg_grid)).build()
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction",
                                     metricName="mse")
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator,
                        numFolds=k, seed=0, collectSubModels=False)
    model = cv.fit(dft)
    return {"best_regParam": float(model.bestModel._java_obj.getRegParam()),
            "avg_metrics_per_grid_point": list(model.avgMetrics),
            "best_MSE": float(min(model.avgMetrics)),
            "grid": list(reg_grid),
            "n_folds": k}


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("cv").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        import random
        random.seed(41)
        rows = []
        for _ in range(500):
            x1 = random.gauss(0, 1); x2 = random.gauss(0, 1)
            y = 1.5 + 0.8 * x1 - 0.3 * x2 + random.gauss(0, 0.5)
            rows.append((x1, x2, y))
        df = spark.createDataFrame(rows, ["x1", "x2", "y"])
        for k, v in cv_linear_regression(df, ["x1", "x2"], "y", k=5).items():
            print(f"  {k:24s}: {v}")
    finally:
        spark.stop()
