# Canary deployment (Reference Ch 32 MLOps)
# Native R via plumber routing; native platforms in Kubernetes ecosystems.
# Run with:  Rscript canary_deployment.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  plumber                     -- REST endpoint routing with traffic weighting\n")
  cat("  vetiver                     -- MLOps helpers around plumber\n")
  cat("Python:\n")
  cat("  seldon-core, kserve         -- native canary deployments in Kubernetes\n")
  cat("  argo-rollouts               -- progressive rollouts on generic pods\n")
  cat("  istio VirtualService        -- traffic-percentage routing at service mesh level\n")
  cat("  bentoml, mlflow             -- side-by-side model deployment\n")
  cat("Refs: Chip Huyen (2022) 'Designing Machine Learning Systems', O'Reilly,\n")
  cat("      ch. 9 (shadow, A-B, canary, interleaved traffic).\n")
}
