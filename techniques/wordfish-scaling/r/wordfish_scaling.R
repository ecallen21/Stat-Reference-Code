# Wordfish / Wordscores scaling (Reference Sec 42.18)
# Native R via quanteda.textmodels; Python custom.
# Run with:  Rscript wordfish_scaling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quanteda.textmodels::textmodel_wordfish -- unsupervised Poisson scaling\n")
  cat("  quanteda.textmodels::textmodel_wordscores -- supervised with reference texts\n")
  cat("Python:\n")
  cat("  custom (Poisson alternating IRLS)\n")
  cat("  textacy, gensim, bertopic (adjacent methods)\n")
  cat("Refs: Slapin & Proksch (2008) 'A scaling model for estimating time-series party\n")
  cat("      positions from texts', AJPS; Laver, Benoit & Garry (2003) 'Extracting\n")
  cat("      policy positions from political texts using words as data', APSR.\n")
}
