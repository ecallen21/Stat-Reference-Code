# ADASYN Oversampling (Reference Sec 47.248)
# Native R via smotefamily; Python via imbalanced-learn / from-scratch.
# Run with:  Rscript adasyn_oversampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  smotefamily::ADAS       -- adaptive synthetic sampling\n")
  cat("  imbalance::adas         -- alternative implementation\n")
  cat("  UBL::AdasynClassif      -- classifier-side ADASYN\n")
  cat("Python:\n")
  cat("  imbalanced-learn.over_sampling.ADASYN\n")
  cat("  From-scratch (see adasyn_oversampling.py)\n")
  cat("Refs: He, H., Bai, Y., Garcia, E.A. & Li, S. (2008) 'ADASYN: Adaptive\n")
  cat("      Synthetic Sampling Approach for Imbalanced Learning',\n")
  cat("      IEEE IJCNN, pp. 1322-1328.\n")
}
