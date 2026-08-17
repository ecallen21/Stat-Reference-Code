# Class imbalance handling (Reference §26.17)
# R via ROSE, DMwR2 (SMOTE), or built-in class weights in glm().
# Run with:  Rscript class_imbalance.R

if (sys.nframe() == 0) {
  cat("Recommended R packages:\n")
  cat("  DMwR2::SMOTE            -- synthetic minority oversampling\n")
  cat("  ROSE::ovun.sample       -- over/under-sampling and ROSE\n")
  cat("  themis:: (tidymodels)   -- SMOTE + variants\n")
  cat("  glm(..., weights = ...) -- class-weighted logistic\n")
}
