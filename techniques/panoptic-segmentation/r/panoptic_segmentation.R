# Panoptic Segmentation (Reference Sec 47.303)
# Native R via reticulate + mmsegmentation; Python via detectron2 / panopticapi / from-scratch.
# Run with:  Rscript panoptic_segmentation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + detectron2 / mmsegmentation panoptic configs\n")
  cat("Python:\n")
  cat("  detectron2 PanopticFPN / Mask2Former (Cheng et al 2022)\n")
  cat("  mmsegmentation MaskFormer / K-Net\n")
  cat("  panopticapi (COCO PQ evaluator)\n")
  cat("  From-scratch numpy (see panoptic_segmentation.py)\n")
  cat("Refs: Kirillov, A., He, K., Girshick, R., Rother, C. & Dollar, P.\n")
  cat("      (2019) 'Panoptic segmentation', CVPR.\n")
}
