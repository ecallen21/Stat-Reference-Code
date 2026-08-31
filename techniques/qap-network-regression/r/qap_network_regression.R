# QAP network regression (Reference Sec 30.6)
# Native R via sna / statnet; Python via netperm.
# Run with:  Rscript qap_network_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  sna::netlm                   -- QAP + double-semi-partial regression\n")
  cat("  statnet suite                 -- broader network analysis + tests\n")
  cat("  asnipe                        -- adjacent for animal-behaviour QAP\n")
  cat("Python:\n")
  cat("  netperm                       -- MRQAP and Mantel permutation tests\n")
  cat("  networkx + custom permutation loop\n")
  cat("Refs: Krackhardt, D. (1988) 'Predicting with networks: nonparametric multiple\n")
  cat("      regression analyses of dyadic data', Social Networks;\n")
  cat("      Dekker, D., Krackhardt, D. & Snijders, T.A.B. (2007) 'Sensitivity of MRQAP\n")
  cat("      tests to collinearity and autocorrelation conditions', Psychometrika.\n")
}
