# One-class SVM (Reference §21.x extra)
# R via e1071 or kernlab.
# Run with:  Rscript one_class_svm.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  e1071::svm(x, type='one-classification', nu=0.05, kernel='radial')\n")
  cat("  kernlab::ksvm(x, type='one-svc', kpar=list(sigma=0.1), nu=0.05)\n")
  cat("  dbscan::lof(x, minPts=k)                -- local outlier factor\n")
  cat("  isotree::isolation.forest / IsolationForest (Python)  -- alternative anomaly detector\n")
  cat("  spatstat / MASS::kde2d for KDE-density anomaly scoring\n")
}
