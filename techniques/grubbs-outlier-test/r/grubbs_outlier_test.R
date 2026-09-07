# Grubbs' + Rosner ESD outlier tests (Reference Sec 3.26)
# Native R via outliers / EnvStats; Python via scipy / from-scratch.
# Run with:  Rscript grubbs_outlier_test.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  outliers::grubbs.test  -- classical Grubbs (one- and two-sided)\n")
  cat("  outliers::dixon.test   -- Dixon Q for small n\n")
  cat("  EnvStats::rosnerTest    -- Rosner extreme-studentised deviate (up to k)\n")
  cat("  car::outlierTest        -- Bonferroni-adjusted studentised residuals\n")
  cat("Python:\n")
  cat("  scipy.stats + custom (see grubbs_outlier_test.py)\n")
  cat("  pyod (multivariate outlier detection ensembles)\n")
  cat("Refs: Grubbs, F.E. (1950) 'Sample criteria for testing outlying\n")
  cat("      observations', Ann Math Stat 21(1): 27-58; Rosner, B. (1983)\n")
  cat("      'Percentage points for a generalized ESD many-outlier procedure',\n")
  cat("      Technometrics 25(2): 165-172.\n")
}
