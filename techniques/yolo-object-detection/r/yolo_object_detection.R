# YOLO Object Detection (Reference Sec 47.295)
# Native R via reticulate + ultralytics; Python via ultralytics / mmdetection / from-scratch.
# Run with:  Rscript yolo_object_detection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + ultralytics       -- YOLOv8/v11 Python via reticulate\n")
  cat("  torch (R) with custom head     -- possible but rare\n")
  cat("Python:\n")
  cat("  ultralytics (YOLOv8, v11)      -- de-facto standard\n")
  cat("  darknet (Redmon original)\n")
  cat("  mmdetection YOLOX / YOLOv3\n")
  cat("  From-scratch (see yolo_object_detection.py)\n")
  cat("Refs: Redmon, J., Divvala, S., Girshick, R. & Farhadi, A. (2016)\n")
  cat("      'You only look once: Unified, real-time object detection', CVPR.\n")
}
