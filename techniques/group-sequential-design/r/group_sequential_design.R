# Group Sequential Design (Reference Sec 47.257)
# Native R via gsDesign / rpact; Python via scipy from-scratch.
# Run with:  Rscript group_sequential_design.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gsDesign::gsDesign      -- Pocock, O'Brien-Fleming, WT boundaries\n")
  cat("  rpact::getDesignGroupSequential -- comprehensive interim planning\n")
  cat("  ldbounds                -- Lan-DeMets alpha-spending boundaries\n")
  cat("  clinfun::gsdesign       -- older but standard implementation\n")
  cat("Python:\n")
  cat("  rpact via reticulate\n")
  cat("  From-scratch simulation (see group_sequential_design.py)\n")
  cat("Refs: Pocock, S.J. (1977) 'Group sequential methods in the design and\n")
  cat("      analysis of clinical trials', Biometrika 64(2), 191-199;\n")
  cat("      O'Brien, P.C. & Fleming, T.R. (1979) 'A multiple testing procedure\n")
  cat("      for clinical trials', Biometrics 35(3), 549-556.\n")
}
