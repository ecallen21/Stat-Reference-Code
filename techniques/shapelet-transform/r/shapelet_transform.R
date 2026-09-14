# Shapelet Transform (Reference Sec 47.292)
# Native R via reticulate + sktime; Python via sktime / tslearn / from-scratch.
# Run with:  Rscript shapelet_transform.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  reticulate + sktime.shapelets   -- Python-native via reticulate\n")
  cat("  tsmp (partial shapelet support)\n")
  cat("Python:\n")
  cat("  sktime.transformations.panel.shapelets.ShapeletTransform\n")
  cat("  tslearn.shapelets.LearningShapelets (LTS)\n")
  cat("  From-scratch numpy (see shapelet_transform.py)\n")
  cat("Refs: Ye, L. & Keogh, E. (2009) 'Time series shapelets: A new primitive\n")
  cat("      for data mining', KDD;  Hills, J., Lines, J., Baranauskas, E.,\n")
  cat("      Mapp, J. & Bagnall, A. (2014) 'Classification of time series by\n")
  cat("      shapelet transformation', Data Min Knowl Discov 28(4), 851-881.\n")
}
