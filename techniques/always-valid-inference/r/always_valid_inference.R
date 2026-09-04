# Always-valid inference (Reference Sec 44.4, 44.12)
# Native R via gsDesign / rpact / ldbounds; Python custom.
# Run with:  Rscript always_valid_inference.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gsDesign::gsDesign               -- group-sequential design + boundaries\n")
  cat("  rpact                             -- adaptive designs including always-valid\n")
  cat("  ldbounds                          -- Lan-DeMets alpha-spending\n")
  cat("Python:\n")
  cat("  sequential-testing                -- mSPRT / anytime-valid utilities\n")
  cat("  confidence-sequence                -- Howard-Ramdas CS\n")
  cat("  custom (scipy.stats + numpy)\n")
  cat("Refs: Johari, Koomen, Pekelis & Walsh (2017) 'Peeking at A/B tests', KDD;\n")
  cat("      Howard, Ramdas, McAuliffe & Sekhon (2021) 'Time-uniform, nonparametric,\n")
  cat("      nonasymptotic confidence sequences', Ann Statist.\n")
}
