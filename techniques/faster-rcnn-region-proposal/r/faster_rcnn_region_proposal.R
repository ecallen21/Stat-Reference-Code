# Faster R-CNN RPN (Reference Sec 47.296)
# Native R via reticulate + torchvision / mmdetection; Python via torchvision / from-scratch.
# Run with:  Rscript faster_rcnn_region_proposal.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + torchvision.models.detection.faster_rcnn\n")
  cat("  torch (R) with custom RPN (rare)\n")
  cat("Python:\n")
  cat("  torchvision.models.detection.faster_rcnn / mask_rcnn\n")
  cat("  mmdetection.detectors.FasterRCNN\n")
  cat("  detectron2.modeling.FasterRCNN\n")
  cat("  From-scratch (see faster_rcnn_region_proposal.py)\n")
  cat("Refs: Ren, S., He, K., Girshick, R. & Sun, J. (2015) 'Faster R-CNN:\n")
  cat("      Towards real-time object detection with region proposal networks',\n")
  cat("      NIPS.\n")
}
