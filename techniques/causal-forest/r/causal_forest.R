# Causal forest -- Athey-Wager GRF (Reference Sec 15.37)
# Native R via grf; Python via econml.
# Run with:  Rscript causal_forest.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  grf::causal_forest             -- reference GRF implementation\n")
  cat("  grf::instrumental_forest       -- IV variant\n")
  cat("  grf::quantile_forest           -- QRF for quantile prediction\n")
  cat("  policytree                     -- optimal treatment rule search\n")
  cat("Python:\n")
  cat("  econml.grf.CausalForest         -- Microsoft EconML port\n")
  cat("  econml.dml.CausalForestDML     -- doubly-robust CausalForest\n")
  cat("Refs: Athey, S., Tibshirani, J. & Wager, S. (2019) 'Generalized random\n")
  cat("      forests', Ann Statist 47(2): 1148-1178; Wager, S. & Athey, S.\n")
  cat("      (2018) 'Estimation and inference of heterogeneous treatment effects\n")
  cat("      using random forests', JASA 113(523): 1228-1242.\n")
}
