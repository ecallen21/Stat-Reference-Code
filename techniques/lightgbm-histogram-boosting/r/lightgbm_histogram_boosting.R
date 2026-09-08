# LightGBM / Histogram GBM (Reference Sec 47.105)
# Native R via lightgbm; Python via lightgbm / sklearn HistGradientBoosting.
# Run with:  Rscript lightgbm_histogram_boosting.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  lightgbm                            -- Microsoft reference GBM\n")
  cat("  xgboost                             -- competing histogram / exact GBM\n")
  cat("  catboost                            -- competing GBM with ordered TS\n")
  cat("Python:\n")
  cat("  lightgbm.LGBMRegressor / LGBMClassifier / LGBMRanker\n")
  cat("  sklearn.ensemble.HistGradientBoostingRegressor / Classifier\n")
  cat("  xgboost with tree_method='hist' or 'gpu_hist'\n")
  cat("  from-scratch                        -- see lightgbm_histogram_boosting.py\n")
  cat("Refs: Ke, Meng, Finley, Wang, Chen, Ma, Ye & Liu (2017) 'LightGBM', NeurIPS.\n")
}
