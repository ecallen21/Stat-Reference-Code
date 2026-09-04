# Transfer entropy (Reference Sec 34.11)
# Native R via RTransferEntropy; Python via PyIF / JIDT.
# Run with:  Rscript transfer_entropy.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  RTransferEntropy              -- Behrendt-Dimpfl-Peter-Zimmermann reference\n")
  cat("  IDPmisc                        -- info-dynamics adjacent\n")
  cat("Python:\n")
  cat("  PyIF                           -- transfer + mutual info\n")
  cat("  IDTxl                          -- multivariate TE + causality\n")
  cat("  JIDT (Java, callable)          -- Lizier reference implementation\n")
  cat("Refs: Schreiber, T. (2000) 'Measuring information transfer', Physical Review Letters;\n")
  cat("      Lizier, J.T. et al. (2011) 'Local measures of information storage in complex\n")
  cat("      distributed computation', Information Sciences.\n")
}
