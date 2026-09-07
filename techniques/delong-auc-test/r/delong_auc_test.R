# DeLong's test for comparing AUCs (Reference Sec 21.3)
# Native R via pROC; Python custom.
# Run with:  Rscript delong_auc_test.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  pROC::roc.test(method='delong')  -- paired / unpaired AUC comparison\n")
  cat("  auc                                 -- alternative implementations\n")
  cat("  riskRegression::Score               -- multi-model discrimination + Brier\n")
  cat("Python:\n")
  cat("  scikit-learn.metrics.roc_auc_score -- AUC only\n")
  cat("  custom (DeLong placement values)\n")
  cat("Refs: DeLong, DeLong & Clarke-Pearson (1988) 'Comparing the areas under two\n")
  cat("      or more correlated receiver operating characteristic curves', Biometrics.\n")
}
