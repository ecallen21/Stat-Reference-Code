# Focal Tversky Loss (Reference Sec 47.299)
# Native R via torch (R) + custom; Python via monai / smp / from-scratch.
# Run with:  Rscript focal_tversky_loss.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (R) with Tversky (alpha, beta) and (1-Tv)^gamma\n")
  cat("  reticulate + segmentation-models-pytorch.losses\n")
  cat("Python:\n")
  cat("  monai.losses.TverskyLoss (Tversky + Focal variants)\n")
  cat("  segmentation-models-pytorch.losses.TverskyLoss / FocalLoss\n")
  cat("  From-scratch numpy (see focal_tversky_loss.py)\n")
  cat("Refs: Salehi, S.S.M., Erdogmus, D. & Gholipour, A. (2017) 'Tversky loss\n")
  cat("      function for image segmentation using 3D fully convolutional deep\n")
  cat("      networks', MICCAI Workshop;  Abraham, N. & Khan, N.M. (2019) 'A\n")
  cat("      novel focal Tversky loss function with improved attention U-Net\n")
  cat("      for lesion segmentation', ISBI.\n")
}
