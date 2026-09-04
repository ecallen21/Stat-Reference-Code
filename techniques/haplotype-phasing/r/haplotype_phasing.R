# Haplotype phasing (Reference Sec 40.27)
# Native R via haplo.stats / gap; production tools SHAPEIT, Beagle (external).
# Run with:  Rscript haplotype_phasing.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  haplo.stats::haplo.em / haplo.glm / haplo.score\n")
  cat("  gap::hap.em                     -- SNP haplotype EM\n")
  cat("  SNPassoc                        -- haplotype-based association\n")
  cat("Python:\n")
  cat("  hail::experimental phase helpers\n")
  cat("  custom EM (Excoffier-Slatkin 1995)\n")
  cat("External:\n")
  cat("  SHAPEIT, Beagle, Eagle          -- large-panel statistical phasing\n")
  cat("Refs: Excoffier & Slatkin (1995) 'Maximum-likelihood estimation of molecular\n")
  cat("      haplotype frequencies in a diploid population', MBE; Stephens, Smith &\n")
  cat("      Donnelly (2001) 'A new statistical method for haplotype reconstruction',\n")
  cat("      AJHG.\n")
}
