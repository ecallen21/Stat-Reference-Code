# Last-layer Bayesian / Neural Linear (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python for the neural feature extractor.
# Run with:  Rscript last_layer_bayesian.R

if (sys.nframe() == 0) {
  cat("R packages: any Bayesian linear regression on frozen features works.\n")
  cat("Python:\n")
  cat("  laplace-torch              -- Kristiadi 2020 last-layer Laplace, torch API\n")
  cat("  sklearn.linear_model.BayesianRidge -- Bayesian LR on penultimate features\n")
  cat("  gpytorch (deep-kernel)     -- GP on the neural feature space\n")
  cat("R alternatives:\n")
  cat("  brms / rstanarm            -- Bayesian LR on features exported from a torch model\n")
  cat("  MASS::mvrnorm              -- sample from posterior N(mu, Sigma) for predictive band\n")
  cat("Refs: Snoek, J. et al. (2015) 'Scalable Bayesian Optimization Using Deep\n")
  cat("      Neural Networks', ICML; Kristiadi, A., Hein, M. & Hennig, P. (2020)\n")
  cat("      'Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks', ICML.\n")
}
