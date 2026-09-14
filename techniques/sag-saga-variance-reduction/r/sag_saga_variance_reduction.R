# SAG / SAGA (Reference Sec 47.337)
# Native R via glmnet / reticulate + sklearn; Python via sklearn / from-scratch.
# Run with:  Rscript sag_saga_variance_reduction.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  glmnet                              -- coord descent; no SAG built-in\n")
  cat("  reticulate + sklearn                -- SAG / SAGA via Python\n")
  cat("  bigMemoryFile / lightning-py wrap  -- large-scale variance-reduced SGD\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.LogisticRegression(solver='saga')\n")
  cat("  lightning.classification.SAGClassifier / SAGAClassifier\n")
  cat("  From-scratch (see sag_saga_variance_reduction.py)\n")
  cat("Refs: Roux, N., Schmidt, M. & Bach, F. (2012) 'A stochastic gradient\n")
  cat("      method with an exponential convergence rate for strongly convex\n")
  cat("      optimization', NeurIPS;  Defazio, A., Bach, F. & Lacoste-Julien, S.\n")
  cat("      (2014) 'SAGA: A fast incremental gradient method with support for\n")
  cat("      non-strongly convex composite objectives', NeurIPS.\n")
}
