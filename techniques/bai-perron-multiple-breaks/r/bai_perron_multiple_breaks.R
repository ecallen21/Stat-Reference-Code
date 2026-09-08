# Bai-Perron multiple structural breaks (Reference Sec 12.17)
# Native R via strucchange; Python via ruptures + from-scratch DP.
# Run with:  Rscript bai_perron_multiple_breaks.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  strucchange::breakpoints  -- Bai-Perron DP + BIC / RSS selection\n")
  cat("  strucchange::sctest       -- Chow / QLR / OLS-MOSUM tests\n")
  cat("  changepoint::cpt.mean       -- PELT / BinSeg segmentations\n")
  cat("Python:\n")
  cat("  ruptures                  -- PELT / BinSeg / dynamic-programming search\n")
  cat("  From-scratch DP (see bai_perron_multiple_breaks.py)\n")
  cat("  statsmodels.stats.diagnostic + custom multi-break search\n")
  cat("Refs: Bai, J. & Perron, P. (1998) 'Estimating and testing linear models\n")
  cat("      with multiple structural changes', Econometrica 66(1); Bai & Perron\n")
  cat("      (2003) 'Computation and analysis of multiple structural change\n")
  cat("      models', J Appl Econometrics 18(1).\n")
}
