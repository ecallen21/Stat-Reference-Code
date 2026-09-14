# FCOS Anchor-Free Detection (Reference Sec 47.302)
# Native R via reticulate + mmdetection; Python via mmdetection / detectron2 / from-scratch.
# Run with:  Rscript anchor_free_fcos.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + mmdetection FCOSHead\n")
  cat("  torch (R) with manual per-point ltrb regression\n")
  cat("Python:\n")
  cat("  mmdetection FCOS / CenterNet / ATSS\n")
  cat("  detectron2.projects.FCOS\n")
  cat("  adelaidet FCOS reference\n")
  cat("  From-scratch (see anchor_free_fcos.py)\n")
  cat("Refs: Tian, Z., Shen, C., Chen, H. & He, T. (2019) 'FCOS: Fully\n")
  cat("      convolutional one-stage object detection', ICCV.\n")
}
