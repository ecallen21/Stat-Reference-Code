# RandAugment / AutoAugment (Reference Sec 47.304)
# Native R via reticulate + torchvision; Python via torchvision / timm / from-scratch.
# Run with:  Rscript randaugment_autoaugment.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + torchvision.transforms.v2 (RandAugment, AutoAugment, TrivialAugment)\n")
  cat("  imager / magick (custom transform sequences)\n")
  cat("Python:\n")
  cat("  torchvision.transforms.v2.RandAugment / AutoAugment / TrivialAugmentWide\n")
  cat("  timm.data.auto_augment (RA + AA + AugMix)\n")
  cat("  albumentations (rich augmentation pipeline)\n")
  cat("  From-scratch (see randaugment_autoaugment.py)\n")
  cat("Refs: Cubuk, E.D., Zoph, B., Mane, D., Vasudevan, V. & Le, Q.V. (2019)\n")
  cat("      'AutoAugment: Learning augmentation strategies from data', CVPR;\n")
  cat("      Cubuk, E.D., Zoph, B., Shlens, J. & Le, Q.V. (2020) 'RandAugment:\n")
  cat("      Practical automated data augmentation with a reduced search space',\n")
  cat("      NeurIPS Workshop / CVPR-W.\n")
}
