# SMOTE Oversampling (Reference Sec 47.247)
# Native R via smotefamily / DMwR / performanceEstimation; Python via imbalanced-learn / from-scratch.
# Run with:  Rscript smote_oversampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  smotefamily::SMOTE         -- classical SMOTE + variants (ADAS, DBSMOTE, ANS)\n")
  cat("  DMwR::SMOTE                -- data-mining book SMOTE\n")
  cat("  performanceEstimation      -- SMOTE wrappers within resampling workflows\n")
  cat("  UBL (Utility-Based Learning) -- SMOTE for regression too\n")
  cat("Python:\n")
  cat("  imbalanced-learn.over_sampling.SMOTE\n")
  cat("  imbalanced-learn.SMOTENC / SMOTEN (categorical extensions)\n")
  cat("  From-scratch (see smote_oversampling.py)\n")
  cat("Refs: Chawla, N.V., Bowyer, K.W., Hall, L.O. & Kegelmeyer, W.P. (2002)\n")
  cat("      'SMOTE: Synthetic Minority Over-sampling Technique', JAIR 16.\n")
}
