# Normality tests (Reference Sec 3.24, 3.25)
# Native R via stats + nortest; Python scipy.stats.
# Run with:  Rscript normality_tests.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::shapiro.test              -- Shapiro-Wilk (n up to 5000)\n")
  cat("  nortest::ad.test / lillie.test / cvm.test / pearson.test\n")
  cat("  tseries::jarque.bera.test        -- moment-based JB test\n")
  cat("Python:\n")
  cat("  scipy.stats.shapiro / anderson / jarque_bera / kstest\n")
  cat("  statsmodels.stats.diagnostic.lilliefors -- KS with estimated params\n")
  cat("Refs: Shapiro & Wilk (1965) 'An analysis of variance test for normality',\n")
  cat("      Biometrika; Stephens (1974) 'EDF statistics for goodness of fit and\n")
  cat("      some comparisons', JASA; Jarque & Bera (1980) 'Efficient tests for\n")
  cat("      normality', Econ Lett.\n")
}
