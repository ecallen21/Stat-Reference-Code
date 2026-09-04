# Hardy-Weinberg equilibrium (Reference Sec 40.25)
# Native R via HardyWeinberg; Python scikit-allel + custom.
# Run with:  Rscript hardy_weinberg.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  HardyWeinberg::HWExact, HWExactStats -- Wigginton exact + mid-p variants\n")
  cat("  genetics::HWE.test              -- chi^2 + exact\n")
  cat("  pegas::hw.test                  -- population-genetics HWE\n")
  cat("Python:\n")
  cat("  scikit-allel::hardy_weinberg_test\n")
  cat("  hail::hl.hardy_weinberg_test    -- distributed HWE for GWAS\n")
  cat("  custom scipy chi2 + exact enumeration\n")
  cat("Refs: Wigginton, Cutler & Abecasis (2005) 'A note on exact tests of Hardy-\n")
  cat("      Weinberg equilibrium', AJHG; Graffelman & Moreno (2013) 'The mid p-value\n")
  cat("      in exact tests for HWE', Stat Appl Genet Mol Biol.\n")
}
