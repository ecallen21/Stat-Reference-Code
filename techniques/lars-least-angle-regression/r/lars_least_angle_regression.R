# LARS (Reference Sec 47.122)
# Native R via lars; Python via sklearn.
# Run with:  Rscript lars_least_angle_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  lars                       -- Efron et al reference implementation\n")
  cat("  glmnet                     -- coord descent lasso path (competitor)\n")
  cat("  elasticnet                 -- LARS-EN for elastic net\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.Lars / LarsCV / LassoLars / lars_path\n")
  cat("  celer                      -- fast coord descent LASSO path\n")
  cat("  from-scratch               -- see lars_least_angle_regression.py\n")
  cat("Refs: Efron, Hastie, Johnstone & Tibshirani (2004) 'Least angle\n")
  cat("      regression', Ann Stat 32(2).\n")
}
