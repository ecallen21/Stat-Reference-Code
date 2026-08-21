# Partial Credit Model + GPCM (Reference §22.8)
# R via ltm::gpcm, mirt::mirt(itemtype = 'gpcm' or 'PCM').
# Run with:  Rscript partial_credit_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ltm::gpcm(Y)                              -- generalized PCM\n")
  cat("  mirt::mirt(Y, 1, itemtype = 'gpcm')       -- comprehensive\n")
  cat("  eRm::PCM(Y)                                -- pure Rasch-family PCM (CML)\n")
}
