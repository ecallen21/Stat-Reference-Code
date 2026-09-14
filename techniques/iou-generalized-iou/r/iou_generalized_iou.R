# IoU / GIoU / DIoU / CIoU (Reference Sec 47.293)
# Native R via torch (R) / custom; Python via torchvision / from-scratch.
# Run with:  Rscript iou_generalized_iou.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (R) - use nnf functions, or manual as in torchvision.ops\n")
  cat("  reticulate + torchvision.ops for a Python wrapper\n")
  cat("Python:\n")
  cat("  torchvision.ops.box_iou / generalized_box_iou / distance_box_iou / complete_box_iou\n")
  cat("  mmcv.ops.bbox_overlaps (variants)\n")
  cat("  From-scratch (see iou_generalized_iou.py)\n")
  cat("Refs: Rezatofighi, H. et al (2019) 'Generalized intersection over union',\n")
  cat("      CVPR;  Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R. & Ren, D. (2020)\n")
  cat("      'Distance-IoU loss: Faster and better learning for bounding box\n")
  cat("      regression', AAAI.\n")
}
