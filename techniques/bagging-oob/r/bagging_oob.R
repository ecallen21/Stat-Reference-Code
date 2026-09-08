# Bagging + OOB (Reference Sec 47.58)
# Native R via ipred / randomForest; Python via sklearn.
# Run with:  Rscript bagging_oob.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ipred::bagging / errorest       -- classical bagging + OOB error\n")
  cat("  randomForest::randomForest      -- includes bagging (mtry = p)\n")
  cat("  caret::train(method='treebag')  -- workflow wrapper\n")
  cat("Python:\n")
  cat("  sklearn.ensemble.BaggingRegressor / BaggingClassifier + oob_score\n")
  cat("  sklearn.ensemble.RandomForestRegressor with max_features=None\n")
  cat("  from-scratch                    -- see bagging_oob.py\n")
  cat("Refs: Breiman (1996) 'Bagging predictors', Machine Learning 24(2);\n")
  cat("      Breiman (2001) 'Random forests', Machine Learning 45(1).\n")
}
