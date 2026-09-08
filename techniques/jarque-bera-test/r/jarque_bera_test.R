# Jarque-Bera normality test (Reference Sec 47.103)
# Native R via tseries / moments; Python via statsmodels / scipy.
# Run with:  Rscript jarque_bera_test.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  tseries::jarque.bera.test    -- reference JB with chi^2 p-value\n")
  cat("  moments::jarque.test         -- lightweight standalone JB\n")
  cat("  normtest / nortest           -- collection of normality tests\n")
  cat("Python:\n")
  cat("  scipy.stats.jarque_bera      -- returns statistic + chi^2 p-value\n")
  cat("  statsmodels.stats.stattools.jarque_bera  -- same, verbose report\n")
  cat("  from-scratch                 -- see jarque_bera_test.py\n")
  cat("Refs: Jarque & Bera (1980) Econ Letters 6(3); Doornik & Hansen (2008)\n")
  cat("      Oxford Bull Econ Stat 70(s1).\n")
}
