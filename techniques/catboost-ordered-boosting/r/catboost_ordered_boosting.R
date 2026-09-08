# CatBoost / Ordered Boosting (Reference Sec 47.88)
# Native R via catboost package; Python via catboost / sklearn.
# Run with:  Rscript catboost_ordered_boosting.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  catboost                     -- R wrapper of CatBoost C++ engine\n")
  cat("  lightgbm                     -- competing GBM with categorical handling\n")
  cat("  xgboost                      -- classic GBM baseline\n")
  cat("Python:\n")
  cat("  catboost.CatBoostRegressor / CatBoostClassifier\n")
  cat("  lightgbm / xgboost           -- alternative GBMs; LGB has cat_ features\n")
  cat("  category_encoders            -- 15+ leakage-aware encoders as alternatives\n")
  cat("  from-scratch                 -- see catboost_ordered_boosting.py\n")
  cat("Refs: Prokhorenkova, Gusev, Vorobev, Dorogush & Gulin (2018) 'CatBoost:\n")
  cat("      unbiased boosting with categorical features', NeurIPS.\n")
}
