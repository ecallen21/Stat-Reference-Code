# Model registry / versioning (Reference Ch 32 MLOps)
# R via reticulate + Python; MLflow client and pins/vetiver are native R options.
# Run with:  Rscript model_registry_versioning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mlflow (R client)           -- native R interface to MLflow model registry\n")
  cat("  vetiver + pins              -- versioned R / Python model store with staging\n")
  cat("  workflowsets (tidymodels)   -- model comparison workflows\n")
  cat("Python:\n")
  cat("  mlflow.tracking.MlflowClient        -- reference open-source model registry\n")
  cat("  wandb Model Registry, neptune model  -- hosted registries\n")
  cat("  sagemaker.model_metrics.ModelPackage -- AWS SageMaker Model Registry\n")
  cat("  vertex-ai Model Registry (google)    -- GCP model registry\n")
  cat("Refs: Zaharia, M. et al. (2018) 'Accelerating the Machine Learning Lifecycle\n")
  cat("      with MLflow', IEEE Data Eng Bulletin.\n")
}
