# U-Net Segmentation (Reference Sec 47.297)
# Native R via reticulate + monai / smp; Python via segmentation-models-pytorch / MONAI / from-scratch.
# Run with:  Rscript u_net_segmentation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + segmentation-models-pytorch\n")
  cat("  torch (R) with manual encoder-decoder + skip connections\n")
  cat("  keras3 (R) - keras.applications + custom U-Net\n")
  cat("Python:\n")
  cat("  segmentation-models-pytorch.Unet / Unet++\n")
  cat("  monai.networks.nets.UNet (biomedical)\n")
  cat("  keras-cv-attention-models\n")
  cat("  From-scratch numpy (see u_net_segmentation.py)\n")
  cat("Refs: Ronneberger, O., Fischer, P. & Brox, T. (2015) 'U-Net:\n")
  cat("      Convolutional networks for biomedical image segmentation', MICCAI.\n")
}
