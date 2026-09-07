# Chow / QLR structural-break test (Reference Sec 12.16)
# Native R via strucchange; Python via statsmodels + custom.
# Run with:  Rscript chow_test_structural_break.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  strucchange::sctest       -- Chow / QLR / OLS-CUSUM / MOSUM / RE\n")
  cat("  strucchange::breakpoints  -- Bai-Perron multiple structural breaks\n")
  cat("  lmtest::sctest             -- alt structural-change tests\n")
  cat("Python:\n")
  cat("  statsmodels.stats.diagnostic.breaks_cusumolsresid\n")
  cat("  From-scratch scipy (see chow_test_structural_break.py)\n")
  cat("  ruptures  -- multiple change-point detection (mean / slope models)\n")
  cat("Refs: Chow, G.C. (1960) 'Tests of equality between sets of coefficients\n")
  cat("      in two linear regressions', Econometrica 28(3): 591-605; Andrews,\n")
  cat("      D.W.K. (1993) 'Tests for parameter instability and structural\n")
  cat("      change with unknown change point', Econometrica 61(4): 821-856.\n")
}
