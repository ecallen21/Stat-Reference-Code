# Experiment tracking (Reference Ch 32 MLOps)
# Native R via reticulate + Python; a few R packages exist.
# Run with:  Rscript experiment_tracking.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mlflow (R client)           -- log runs / metrics / artifacts from R\n")
  cat("  wandb (R via reticulate)    -- W&B run wrapper\n")
  cat("  vetiver + pins              -- store versioned models + metadata\n")
  cat("Python:\n")
  cat("  mlflow                       -- open-source reference tracking + registry\n")
  cat("  wandb, neptune, comet, aim   -- hosted / open experiment trackers\n")
  cat("  dvc                          -- content-addressed data + artifact versioning\n")
  cat("Refs: Zaharia, M. et al. (2018) 'Accelerating the Machine Learning Lifecycle\n")
  cat("      with MLflow', IEEE Data Eng Bulletin;\n")
  cat("      Sculley, D. et al. (2015) 'Hidden Technical Debt in ML Systems', NeurIPS.\n")
}
