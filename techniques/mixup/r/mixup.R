# Mixup (Reference Ch 30 Robustness)
# R via reticulate + Python; a few lines of native R also suffice.
# Run with:  Rscript mixup.R

if (sys.nframe() == 0) {
  cat("R packages: mixup is a small data-augmentation loop; native R is easy.\n")
  cat("  keras3 / tensorflow (R)     -- pass mixed batches through fit()\n")
  cat("  torch (R port)              -- manual mixup in the dataloader collate_fn\n")
  cat("Python:\n")
  cat("  torchvision.transforms.v2.MixUp\n")
  cat("  timm (Wightman) mixup / cutmix helpers\n")
  cat("  augly, kornia               -- broader data augmentation toolkits\n")
  cat("Refs: Zhang, H., Cisse, M., Dauphin, Y. & Lopez-Paz, D. (2018)\n")
  cat("      'mixup: Beyond Empirical Risk Minimization', ICLR.\n")
}
