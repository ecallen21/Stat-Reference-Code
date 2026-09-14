# Cook's Distance / Leverage (Reference Sec 47.333)
# Native R via stats::influence.measures / car; Python via statsmodels / from-scratch.
# Run with:  Rscript cook_distance_leverage.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::influence.measures    -- Cook D, hat, DFBETAS, DFFITS, cov ratio\n")
  cat("  car::influenceIndexPlot      -- diagnostic plots\n")
  cat("  DHARMa                        -- randomised quantile residuals\n")
  cat("  performance::check_outliers  -- easystats convenience wrapper\n")
  cat("Python:\n")
  cat("  statsmodels.stats.outliers_influence.OLSInfluence\n")
  cat("  yellowbrick.regressor.CooksDistance\n")
  cat("  From-scratch (see cook_distance_leverage.py)\n")
  cat("Refs: Cook, R.D. (1977) 'Detection of influential observations in\n")
  cat("      linear regression', Technometrics 19(1);\n")
  cat("      Belsley, D.A., Kuh, E. & Welsch, R.E. (1980) 'Regression\n")
  cat("      Diagnostics', Wiley.\n")
}
