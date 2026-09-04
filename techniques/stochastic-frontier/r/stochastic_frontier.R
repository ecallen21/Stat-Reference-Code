# Stochastic frontier analysis (Reference Sec 35.22)
# Native R via frontier; Python via pysfa.
# Run with:  Rscript stochastic_frontier.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  frontier                     -- Coelli-Perelman-Battese reference (SFA + DEA-adjacent)\n")
  cat("  Benchmarking                  -- DEA + SFA + Malmquist\n")
  cat("  sfaR                          -- broader SFA family (heterogeneous inefficiency)\n")
  cat("Python:\n")
  cat("  pysfa                          -- Battese-Coelli style SFA\n")
  cat("  statsmodels + MLE loop         -- manual (this demo)\n")
  cat("Refs: Aigner, D., Lovell, C.A.K. & Schmidt, P. (1977) 'Formulation and\n")
  cat("      estimation of stochastic frontier production function models',\n")
  cat("      J Econometrics; Meeusen, W. & van den Broeck, J. (1977) 'Efficiency\n")
  cat("      estimation from Cobb-Douglas production functions with composed error',\n")
  cat("      Int Econ Rev.\n")
}
