# DETR (Reference Sec 47.301)
# Native R via reticulate + transformers; Python via transformers / detectron2 / from-scratch.
# Run with:  Rscript detr_transformer_detection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + transformers.DetrForObjectDetection\n")
  cat("  torch (R) with manual encoder-decoder\n")
  cat("Python:\n")
  cat("  transformers.DetrModel / DetrForObjectDetection\n")
  cat("  facebookresearch/detr (original)\n")
  cat("  mmdetection DETR / Deformable DETR / DINO\n")
  cat("  From-scratch scipy (see detr_transformer_detection.py)\n")
  cat("Refs: Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A. &\n")
  cat("      Zagoruyko, S. (2020) 'End-to-end object detection with\n")
  cat("      transformers', ECCV.\n")
}
