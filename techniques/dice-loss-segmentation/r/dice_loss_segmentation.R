# Dice Loss (Reference Sec 47.298)
# Native R via torch (R) + custom; Python via monai / smp / from-scratch.
# Run with:  Rscript dice_loss_segmentation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (R) with dice_coef() and 1 - dice loss\n")
  cat("  reticulate + segmentation-models-pytorch.losses.DiceLoss\n")
  cat("Python:\n")
  cat("  monai.losses.DiceLoss / DiceCELoss\n")
  cat("  segmentation-models-pytorch.losses.DiceLoss\n")
  cat("  From-scratch numpy (see dice_loss_segmentation.py)\n")
  cat("Refs: Milletari, F., Navab, N. & Ahmadi, S.-A. (2016) 'V-Net: Fully\n")
  cat("      convolutional neural networks for volumetric medical image\n")
  cat("      segmentation', 3DV;  Sudre, C.H. et al (2017) 'Generalised Dice\n")
  cat("      overlap as a deep learning loss function for highly unbalanced\n")
  cat("      segmentations', DLMIA.\n")
}
