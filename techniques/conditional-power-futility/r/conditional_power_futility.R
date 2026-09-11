# Conditional Power / Futility (Reference Sec 47.259)
# Native R via rpact / gsDesign; Python via scipy from-scratch.
# Run with:  Rscript conditional_power_futility.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rpact::getConditionalPower  -- CP at interim looks\n")
  cat("  gsDesign::gsCP              -- CP under specified drift\n")
  cat("  gsDesign::gsBoundSummary    -- table of stopping probabilities\n")
  cat("Python:\n")
  cat("  rpact via reticulate\n")
  cat("  From-scratch (see conditional_power_futility.py)\n")
  cat("Refs: Lan, K.K.G. & Wittes, J. (1988) 'The B-value: A tool for\n")
  cat("      monitoring data', Biometrics 44(2), 579-585.\n")
}
