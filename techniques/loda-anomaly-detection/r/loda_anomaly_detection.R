# LODA (Reference Sec 47.135)
# Python via pyod.
# Run with:  Rscript loda_anomaly_detection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class LODA in R -- roll your own or reticulate pyod\n")
  cat("  isotree                     -- isolation-forest alternative\n")
  cat("  dbscan                      -- density-based anomaly cousin\n")
  cat("Python:\n")
  cat("  pyod.models.LODA            -- reference implementation\n")
  cat("  pyod.models.IForest / OCSVM / KDE / COPOD  -- ensemble AD stack\n")
  cat("  sklearn.ensemble.IsolationForest\n")
  cat("  from-scratch                -- see loda_anomaly_detection.py\n")
  cat("Refs: Pevny (2016) 'Loda: Lightweight on-line detector of anomalies',\n")
  cat("      Machine Learning 102(2).\n")
}
