# Feature squeezing (Reference Ch 30 Robustness)
# R via reticulate + Python; native R can do bit-depth and median easily.
# Run with:  Rscript feature_squeezing.R

if (sys.nframe() == 0) {
  cat("R packages: bit-depth and median filters are one-liners in native R.\n")
  cat("  imager                       -- median_blur, quantise\n")
  cat("  EBImage (Bioconductor)       -- medianFilter + intensity quantisation\n")
  cat("Python:\n")
  cat("  scipy.ndimage.median_filter\n")
  cat("  skimage.filters.median, skimage.exposure.rescale_intensity\n")
  cat("  foolbox / advertorch         -- preprocessing defence wrappers\n")
  cat("Refs: Xu, W., Evans, D. & Qi, Y. (2018) 'Feature Squeezing: Detecting\n")
  cat("      Adversarial Examples in Deep Neural Networks', NDSS.\n")
}
