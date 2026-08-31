# Shadow deployment (Reference Ch 32 MLOps)
# Native R for the dual-scoring loop; production platforms in Python.
# Run with:  Rscript shadow_deployment.R

if (sys.nframe() == 0) {
  cat("R packages: shadow scoring is a two-model prediction + logging pattern.\n")
  cat("  plumber                     -- REST endpoint routing / dual-scoring\n")
  cat("  vetiver                     -- MLOps + shadow / staging models via pins\n")
  cat("Python:\n")
  cat("  seldon-core, kserve         -- native shadow deployments in Kubernetes\n")
  cat("  bentoml, mlflow             -- side-by-side model serving + logging\n")
  cat("  ray-serve                    -- multi-model routing with logging\n")
  cat("Refs: Chip Huyen (2022) 'Designing Machine Learning Systems', O'Reilly,\n")
  cat("      ch. 9 'Continual Learning and Test in Production' (shadow / A-B / canary).\n")
}
