# CutMix (Reference Ch 30 Robustness)
# R via reticulate + Python; the patch-swap loop is easy in native R.
# Run with:  Rscript cutmix.R

if (sys.nframe() == 0) {
  cat("R packages: patch-swap is a few array indexes; native R is fine.\n")
  cat("  keras3 (R)                   -- add cutmix in a custom collate\n")
  cat("  torch (R port)               -- dataloader collate_fn cutmix\n")
  cat("Python:\n")
  cat("  torchvision.transforms.v2.CutMix\n")
  cat("  timm (Wightman) mixup / cutmix helpers\n")
  cat("  augly, kornia               -- broader augmentation toolkits\n")
  cat("Refs: Yun, S., Han, D., Chun, S., Oh, S., Yoo, Y. & Choe, J. (2019)\n")
  cat("      'CutMix: Regularization Strategy to Train Strong Classifiers with\n")
  cat("      Localizable Features', ICCV.\n")
}
