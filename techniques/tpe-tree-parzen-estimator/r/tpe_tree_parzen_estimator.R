# Tree-structured Parzen Estimator (Reference Sec 47.278)
# Native R via mlr3mbo; Python via hyperopt / Optuna / from-scratch.
# Run with:  Rscript tpe_tree_parzen_estimator.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mlr3mbo                      -- Bayesian MBO (GP by default; TPE via extension)\n")
  cat("  reticulate::use_python + hyperopt / Optuna\n")
  cat("Python:\n")
  cat("  hyperopt.tpe.suggest         -- original TPE\n")
  cat("  Optuna TPESampler            -- default sampler\n")
  cat("  From-scratch (see tpe_tree_parzen_estimator.py)\n")
  cat("Refs: Bergstra, J., Bardenet, R., Bengio, Y. & Kegl, B. (2011)\n")
  cat("      'Algorithms for hyper-parameter optimization', NeurIPS 24.\n")
}
