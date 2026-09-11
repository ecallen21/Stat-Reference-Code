# Tomek Links Undersampling (Reference Sec 47.249)
# Native R via unbalanced / UBL; Python via imbalanced-learn / from-scratch.
# Run with:  Rscript tomek_links_undersampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  unbalanced::ubTomek     -- Tomek-link removal\n")
  cat("  UBL::TomekClassif       -- Tomek variant in unified framework\n")
  cat("  themis::step_tomek      -- tidymodels recipe step\n")
  cat("Python:\n")
  cat("  imbalanced-learn.under_sampling.TomekLinks\n")
  cat("  imbalanced-learn.combine.SMOTETomek\n")
  cat("  From-scratch (see tomek_links_undersampling.py)\n")
  cat("Refs: Tomek, I. (1976) 'Two Modifications of CNN', IEEE Trans SMC 6(11).\n")
}
