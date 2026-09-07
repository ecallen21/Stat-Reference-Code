# D-optimal design (Reference Sec 17.19)
# Native R via AlgDesign; Python pyDOE2 + custom.
# Run with:  Rscript d_optimal_design.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  AlgDesign::optFederov            -- Fedorov exchange for D / A / I optimality\n")
  cat("  AlgDesign::optBlock              -- blocked D-optimal designs\n")
  cat("  DoE.wrapper                       -- unifies pyDOE-style constructors\n")
  cat("  skpr                              -- power + optimality diagnostics\n")
  cat("Python:\n")
  cat("  pyDOE2                            -- fractional / full factorial helpers\n")
  cat("  dexpy                              -- D-optimal + I-optimal designs\n")
  cat("  custom (numpy + Fedorov)\n")
  cat("Refs: Fedorov (1972) Theory of Optimal Experiments, Academic Press;\n")
  cat("      Atkinson, Donev & Tobias (2007) Optimum Experimental Designs, OUP.\n")
}
