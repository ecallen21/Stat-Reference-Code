# Risk-adjusted control charts (Reference Sec 37.10)
# Native R via vlad; Python via custom.
# Run with:  Rscript risk_adjusted_control_charts.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  vlad                          -- Grigg-Farewell VLAD + risk-adjusted CUSUM\n")
  cat("  runstats                      -- adjacent runs / cumulative helpers\n")
  cat("Python:\n")
  cat("  pyspc + custom                 -- manual\n")
  cat("Refs: Lovegrove, J. et al. (1997) 'Monitoring the results of cardiac surgery',\n")
  cat("      Lancet; Steiner, S.H. et al. (2000) 'Monitoring surgical performance\n")
  cat("      using risk-adjusted cumulative sum charts', Biostatistics.\n")
}
