# XGBoost (Reference Sec 47.104)
# Native R via xgboost; Python via xgboost / lightgbm / catboost.
# Run with:  Rscript xgboost_boosting.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  xgboost                       -- reference GBM implementation\n")
  cat("  xgboost.explainer / SHAPforxgboost -- attribution / interpretability\n")
  cat("  lightgbm / catboost           -- competing GBMs\n")
  cat("Python:\n")
  cat("  xgboost.XGBRegressor / XGBClassifier\n")
  cat("  lightgbm.LGBMRegressor       -- faster on many datasets\n")
  cat("  catboost.CatBoostRegressor   -- best out-of-box categorical handling\n")
  cat("  shap                          -- fast tree SHAP for XGB / LGB / CB\n")
  cat("  from-scratch                  -- see xgboost_boosting.py\n")
  cat("Refs: Chen & Guestrin (2016) 'XGBoost: A scalable tree boosting system', KDD.\n")
}
