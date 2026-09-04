# Penalized regression for clinical prediction (Reference Sec 39.9)
# Native R via glmnet + rms::pentrace; Python sklearn.
# Run with:  Rscript penalized_clinical_prediction.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  glmnet::cv.glmnet              -- ridge, lasso, elastic-net logistic + Cox\n")
  cat("  rms::pentrace                  -- effective-df targeted ridge (Harrell)\n")
  cat("  caret                          -- unified train() interface\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.LogisticRegressionCV / LassoCV / ElasticNetCV\n")
  cat("Refs: Steyerberg, E.W. (2019) Clinical Prediction Models, 2nd ed., Springer, Ch 12;\n")
  cat("      van Houwelingen, J.C. & Le Cessie, S. (1990) 'Predictive value of\n")
  cat("      statistical models', Stat Med; Tibshirani, R. (1996) 'Regression shrinkage\n")
  cat("      and selection via the lasso', JRSS-B.\n")
}
