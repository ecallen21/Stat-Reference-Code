# Exponentiated-gradient reduction for fair classification (Reference Ch 31 Fairness)
# R via reticulate + Python; fairlearn is the reference implementation.
# Run with:  Rscript exponentiated_gradient_reduction.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairml                      -- reduction-style fair SVM / LR\n")
  cat("  mlr3fairness                -- adapters for fairlearn reductions\n")
  cat("Python:\n")
  cat("  fairlearn.reductions.ExponentiatedGradient   (Agarwal 2018 reference)\n")
  cat("  fairlearn.reductions.GridSearch              (grid-search reduction cousin)\n")
  cat("  aif360.algorithms.inprocessing.MetaFair (Celis 2019, alternative reduction)\n")
  cat("Refs: Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J. & Wallach, H. (2018)\n")
  cat("      'A Reductions Approach to Fair Classification', ICML.\n")
}
