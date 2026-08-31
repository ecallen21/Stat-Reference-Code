# Label smoothing (Reference Ch 30 Robustness)
# R via reticulate + Python; trivially implemented in native R too.
# Run with:  Rscript label_smoothing.R

if (sys.nframe() == 0) {
  cat("R packages: label smoothing is a one-line target modification.\n")
  cat("  keras3, tensorflow (R)      -- loss_categorical_crossentropy(label_smoothing = eps)\n")
  cat("  torch (R port)              -- pass label_smoothing = eps to nnf_cross_entropy\n")
  cat("Python:\n")
  cat("  torch.nn.CrossEntropyLoss(label_smoothing = eps)\n")
  cat("  tf.keras.losses.CategoricalCrossentropy(label_smoothing = eps)\n")
  cat("  timm, transformers          -- label_smoothing arg on training args\n")
  cat("Refs: Szegedy, C. et al. (2016) 'Rethinking the Inception Architecture\n")
  cat("      for Computer Vision', CVPR;\n")
  cat("      Muller, R., Kornblith, S. & Hinton, G. (2019)\n")
  cat("      'When Does Label Smoothing Help?', NeurIPS.\n")
}
