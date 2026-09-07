# BART -- Bayesian additive regression trees (Reference Sec 5.16)
# Native R via BART / dbarts / bartMachine; Python via pymc-bart.
# Run with:  Rscript bart_bayesian_additive_regression_trees.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  BART                     -- Chipman et al. reference implementation\n")
  cat("  dbarts                    -- fast BART with unifed API\n")
  cat("  bartMachine               -- Kapelner-Bleich BART with predictor importance\n")
  cat("  BayesTree                 -- original CRAN BART\n")
  cat("  stan4bart                 -- BART + fixed / random effects via Stan\n")
  cat("Python:\n")
  cat("  pymc-bart                 -- PyMC integrated BART\n")
  cat("  bartpy                    -- pure-Python BART\n")
  cat("Refs: Chipman, H.A., George, E.I. & McCulloch, R.E. (2010) 'BART: Bayesian\n")
  cat("      additive regression trees', Ann Appl Stat 4(1): 266-298.\n")
}
