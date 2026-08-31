# Group LASSO (Reference Sec 32.9)
# Native R via grplasso / gglasso; Python via celer / group-lasso.
# Run with:  Rscript group_lasso.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  grplasso                     -- Meier-van-de-Geer-Buhlmann group LASSO\n")
  cat("  gglasso                      -- Yang-Zou group LASSO with more penalties\n")
  cat("  glmnet(family='mgaussian') -- multi-response elastic-net (adjacent)\n")
  cat("Python:\n")
  cat("  celer                          -- fast group LASSO for large-scale problems\n")
  cat("  group-lasso (pip pkg)          -- pure-Python coordinate descent\n")
  cat("  scikit-learn MultiTaskLasso    -- multi-response L21 penalty (adjacent)\n")
  cat("Refs: Yuan, M. & Lin, Y. (2006) 'Model selection and estimation in regression\n")
  cat("      with grouped variables', JRSS-B;\n")
  cat("      Meier, L., van de Geer, S. & Buhlmann, P. (2008) 'The group lasso for\n")
  cat("      logistic regression', JRSS-B.\n")
}
