# Sure Independence Screening (Reference Sec 32.7)
# Native R via SIS package; Python via reticulate.
# Run with:  Rscript sure_independence_screening.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  SIS                          -- Fan-Lv reference: SIS, ISIS, gSIS\n")
  cat("  glmnet + screening loop       -- manual pipeline for GLM screening\n")
  cat("Python:\n")
  cat("  celer                          -- LASSO with active-set screening\n")
  cat("  scikit-learn SelectKBest      -- univariate feature selection (adjacent)\n")
  cat("Refs: Fan, J. & Lv, J. (2008) 'Sure independence screening for ultrahigh\n")
  cat("      dimensional feature space', JRSS-B;\n")
  cat("      Fan, J., Samworth, R. & Wu, Y. (2009) 'Ultrahigh dimensional feature\n")
  cat("      selection: beyond the linear model', JMLR.\n")
}
