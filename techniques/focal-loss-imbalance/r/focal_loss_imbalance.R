# Focal Loss (Reference Sec 47.251)
# Native R via torch / keras3 / mlr3learners; Python via torchvision / from-scratch.
# Run with:  Rscript focal_loss_imbalance.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nnf_binary_cross_entropy + custom focal weighting\n")
  cat("  keras3::loss_binary_focal_crossentropy\n")
  cat("  lightgbm::lgb.train(objective='binary', is_unbalance=TRUE)\n")
  cat("Python:\n")
  cat("  torchvision.ops.sigmoid_focal_loss\n")
  cat("  tf.keras.losses.BinaryFocalCrossentropy\n")
  cat("  From-scratch (see focal_loss_imbalance.py)\n")
  cat("Refs: Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollar, P. (2017)\n")
  cat("      'Focal Loss for Dense Object Detection', IEEE ICCV, 2999-3007.\n")
}
