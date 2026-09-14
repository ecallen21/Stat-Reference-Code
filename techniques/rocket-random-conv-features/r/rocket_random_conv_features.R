# ROCKET (Reference Sec 47.291)
# Native R via mlrocket via reticulate; Python via sktime / tslearn / from-scratch.
# Run with:  Rscript rocket_random_conv_features.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + sktime.rocket   -- Python-native ROCKET via reticulate\n")
  cat("  Rocket via mlr3 (via reticulate wrappers)\n")
  cat("Python:\n")
  cat("  sktime.transformations.panel.rocket.Rocket / MiniRocket / MultiRocket\n")
  cat("  tslearn.transformations.Rocket\n")
  cat("  From-scratch numpy (see rocket_random_conv_features.py)\n")
  cat("Refs: Dempster, A., Petitjean, F. & Webb, G.I. (2020) 'ROCKET:\n")
  cat("      exceptionally fast and accurate time series classification using\n")
  cat("      random convolutional kernels', Data Min Knowl Discov 34, 1454-1495.\n")
}
