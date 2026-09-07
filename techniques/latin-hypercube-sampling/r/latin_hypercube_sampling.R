# Latin hypercube sampling (Reference Sec 45.7)
# Native R via lhs; Python scipy.stats.qmc + pyDOE2.
# Run with:  Rscript latin_hypercube_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  lhs::randomLHS / maximinLHS      -- standard + maximin optimal LHS\n")
  cat("  DiceDesign                        -- LHS + discrepancy criteria\n")
  cat("  SLHD                              -- sliced Latin hypercube design\n")
  cat("Python:\n")
  cat("  scipy.stats.qmc.LatinHypercube   -- SciPy 1.7+ quasi-MC LHS\n")
  cat("  pyDOE2::lhs                       -- classic LHS\n")
  cat("  smt.sampling_methods              -- surrogate modelling toolkit\n")
  cat("Refs: McKay, Beckman & Conover (1979) 'A comparison of three methods for\n")
  cat("      selecting values of input variables in the analysis of output from a\n")
  cat("      computer code', Technometrics; Morris & Mitchell (1995) 'Exploratory\n")
  cat("      designs for computational experiments', JSPI.\n")
}
