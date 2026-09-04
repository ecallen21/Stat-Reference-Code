# Multicollinearity: VIF + condition number (Reference Sec 41.8)
# Native R via car::vif; Python statsmodels + custom.
# Run with:  Rscript multicollinearity_vif.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  car::vif                        -- classical + generalised VIF\n")
  cat("  performance::check_collinearity -- VIF + interpretation\n")
  cat("  caret::findCorrelation          -- prune highly-correlated features\n")
  cat("  perturb::colldiag               -- Belsley-Kuh-Welsch condition indices\n")
  cat("Python:\n")
  cat("  statsmodels.stats.outliers_influence.variance_inflation_factor\n")
  cat("  numpy.linalg.cond               -- condition number of X\n")
  cat("Refs: Dormann et al. (2013) 'Collinearity: a review of methods to deal with\n")
  cat("      it and a simulation study evaluating their performance', Ecography;\n")
  cat("      Kutner, Nachtsheim, Neter & Li (2005) Applied Linear Statistical Models,\n")
  cat("      5th ed., McGraw-Hill, Ch 7 & 10.\n")
}
