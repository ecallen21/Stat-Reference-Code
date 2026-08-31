# Concept-drift detection: ADWIN + DDM (Reference Ch 32 MLOps)
# R via reticulate + Python; native R packages are limited.
# Run with:  Rscript concept_drift_adwin.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  drifter                     -- generic concept-drift dashboards\n")
  cat("  bootcluster                 -- online drift-tolerant clustering\n")
  cat("Python:\n")
  cat("  river.drift.ADWIN / DDM / EDDM / PageHinkley -- reference on-line detectors\n")
  cat("  scikit-multiflow             -- ADWIN, DDM, KSWIN, HDDM implementations\n")
  cat("  alibi-detect                 -- online sequential drift detectors (MMDDrift-Online)\n")
  cat("Refs: Bifet, A. & Gavalda, R. (2007) 'Learning from Time-Changing Data with\n")
  cat("      Adaptive Windowing (ADWIN)', SDM;\n")
  cat("      Gama, J. et al. (2004) 'Learning with Drift Detection (DDM)', SBIA.\n")
}
