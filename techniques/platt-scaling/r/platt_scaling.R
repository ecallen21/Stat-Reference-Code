# Platt Scaling (Reference Sec 47.282)
# Native R via probably / e1071; Python via sklearn / from-scratch.
# Run with:  Rscript platt_scaling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  probably::cal_estimate_logistic -- Platt logistic calibration\n")
  cat("  e1071::probability=TRUE (SVM)   -- built-in Platt scaling\n")
  cat("  tidymodels workflows            -- calibration recipes\n")
  cat("Python:\n")
  cat("  sklearn.calibration.CalibratedClassifierCV(method='sigmoid')\n")
  cat("  netcal.scaling.LogisticCalibration\n")
  cat("  From-scratch scipy (see platt_scaling.py)\n")
  cat("Refs: Platt, J.C. (1999) 'Probabilistic outputs for support vector\n")
  cat("      machines and comparisons to regularized likelihood methods',\n")
  cat("      Advances in Large Margin Classifiers, MIT Press.\n")
}
