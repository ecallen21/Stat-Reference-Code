# Distributional regression (Reference Sec 33.12)
# Native R has multiple strong options; Python via reticulate.
# Run with:  Rscript distributional_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gamlss                       -- parametric distributional regression (see gamlss)\n")
  cat("  bamlss                        -- Bayesian additive models for location/scale/shape\n")
  cat("  drf                           -- Distributional Random Forests (Athey/Wager)\n")
  cat("  quantregForest                -- quantile random forests (Meinshausen)\n")
  cat("Python:\n")
  cat("  ngboost                       -- gradient boosting with distributional heads\n")
  cat("  distributional-forests (drf)  -- Python bindings\n")
  cat("  gluon-ts                      -- distributional forecasts for time series\n")
  cat("Refs: Klein, Kneib, Klasen & Lang (2015) 'Bayesian structured additive\n")
  cat("      distributional regression', Statistical Modelling;\n")
  cat("      Duan, T. et al. (2020) 'NGBoost: Natural Gradient Boosting for\n")
  cat("      Probabilistic Prediction', ICML.\n")
}
