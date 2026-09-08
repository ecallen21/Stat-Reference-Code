# Grad-CAM saliency (Reference Sec 47.114)
# Python via pytorch-grad-cam / captum.
# Run with:  Rscript grad_cam_saliency.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class R port -- reticulate wraps Python libs\n")
  cat("  torch (R)                      -- build Grad-CAM manually with hooks\n")
  cat("Python:\n")
  cat("  pytorch-grad-cam               -- Selvaraju + Grad-CAM++, Score-CAM, XGradCAM\n")
  cat("  captum.attr.GuidedGradCam / LayerGradCam\n")
  cat("  tf-keras-vis                   -- Keras/TensorFlow Grad-CAM\n")
  cat("  iNNvestigate                   -- Keras attribution library\n")
  cat("  from-scratch                   -- see grad_cam_saliency.py\n")
  cat("Refs: Selvaraju et al (2017) 'Grad-CAM', ICCV; Chattopadhay et al (2018)\n")
  cat("      'Grad-CAM++', WACV.\n")
}
