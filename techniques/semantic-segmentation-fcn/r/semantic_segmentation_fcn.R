# FCN Semantic Segmentation (Reference Sec 47.300)
# Native R via reticulate + torchvision; Python via torchvision / mmsegmentation / from-scratch.
# Run with:  Rscript semantic_segmentation_fcn.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + torchvision.models.segmentation.fcn_resnet50\n")
  cat("  torch (R) with manual encoder-decoder + 1x1 conv head\n")
  cat("Python:\n")
  cat("  torchvision.models.segmentation.fcn_resnet50 / fcn_resnet101\n")
  cat("  mmsegmentation FCN configs\n")
  cat("  keras.applications + custom deconv upsampling\n")
  cat("  From-scratch numpy (see semantic_segmentation_fcn.py)\n")
  cat("Refs: Long, J., Shelhamer, E. & Darrell, T. (2015) 'Fully convolutional\n")
  cat("      networks for semantic segmentation', CVPR.\n")
}
