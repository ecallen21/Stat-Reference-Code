# Class-Balanced Loss (Reference Sec 47.252)
# Native R via caret / mlr3 with class weights; Python via manual per-batch weighting.
# Run with:  Rscript class_balanced_loss.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  caret::train(weights = ...)     -- effective-number weights\n")
  cat("  mlr3learners with class.weights -- unified interface\n")
  cat("  xgboost(scale_pos_weight = ...) -- boosting-flavoured rebalancing\n")
  cat("Python:\n")
  cat("  sklearn class_weight='balanced' (inverse-freq)\n")
  cat("  Custom per-class weights in tf/pytorch training loops\n")
  cat("  From-scratch (see class_balanced_loss.py)\n")
  cat("Refs: Cui, Y., Jia, M., Lin, T.-Y., Song, Y. & Belongie, S. (2019)\n")
  cat("      'Class-Balanced Loss Based on Effective Number of Samples',\n")
  cat("      IEEE CVPR, 9268-9277.\n")
}
