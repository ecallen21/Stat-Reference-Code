# Alpha-Spending Function (Reference Sec 47.258)
# Native R via ldbounds / gsDesign / rpact; Python via scipy from-scratch.
# Run with:  Rscript alpha_spending_lan_demets.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ldbounds::ldBounds         -- Lan-DeMets flexible spending\n")
  cat("  gsDesign::sfLDOF / sfLDPocock -- built-in spending functions\n")
  cat("  rpact::getDesignGroupSequential(typeOfDesign='asOF') etc.\n")
  cat("Python:\n")
  cat("  rpact via reticulate\n")
  cat("  From-scratch (see alpha_spending_lan_demets.py)\n")
  cat("Refs: Lan, K.K.G. & DeMets, D.L. (1983) 'Discrete sequential boundaries\n")
  cat("      for clinical trials', Biometrika 70(3), 659-663.\n")
}
