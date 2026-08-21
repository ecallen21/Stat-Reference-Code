# DIF: Mantel-Haenszel + Logistic (Reference §22.11)
# R via difR (Magis).
# Run with:  Rscript dif_mantel_haenszel.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  difR::difMH(Y, group)       -- Mantel-Haenszel DIF\n")
  cat("  difR::difLogistic(Y, group) -- Swaminathan-Rogers logistic DIF\n")
  cat("  difR::difLord               -- IRT-based DIF\n")
}
