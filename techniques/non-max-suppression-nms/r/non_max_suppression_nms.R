# Non-Maximum Suppression (Reference Sec 47.294)
# Native R via reticulate + torchvision; Python via torchvision / mmcv / from-scratch.
# Run with:  Rscript non_max_suppression_nms.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + torchvision.ops.nms\n")
  cat("  imager / EBImage + custom NMS (rare use)\n")
  cat("Python:\n")
  cat("  torchvision.ops.nms / batched_nms\n")
  cat("  mmcv.ops.nms / soft_nms\n")
  cat("  ultralytics.utils.ops.non_max_suppression (YOLOv8)\n")
  cat("  From-scratch (see non_max_suppression_nms.py)\n")
  cat("Refs: Neubeck, A. & Van Gool, L. (2006) 'Efficient non-maximum\n")
  cat("      suppression', ICPR;  Bodla, N., Singh, B., Chellappa, R. &\n")
  cat("      Davis, L.S. (2017) 'Soft-NMS -- improving object detection\n")
  cat("      with one line of code', ICCV.\n")
}
