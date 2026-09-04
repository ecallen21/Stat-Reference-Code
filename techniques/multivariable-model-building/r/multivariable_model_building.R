# Multivariable model building (Reference Sec 39.2)
# Native R via rms; Python sklearn + custom.
# Run with:  Rscript multivariable_model_building.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rms (lrm, fastbw, pentrace, validate) -- Harrell's full-model strategy\n")
  cat("  MASS::stepAIC                  -- stepwise (Harrell warns against for prediction)\n")
  cat("  glmnet                         -- LASSO / ridge / elastic net\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.LogisticRegression / LogisticRegressionCV\n")
  cat("  custom                         -- backward AIC + van Houwelingen shrinkage\n")
  cat("Refs: Harrell, F.E. (2015) Regression Modeling Strategies, 2nd ed., Springer;\n")
  cat("      Steyerberg, E.W. (2019) Clinical Prediction Models, 2nd ed., Springer, Ch 11;\n")
  cat("      van Houwelingen, J.C. & Le Cessie, S. (1990) Stat Med.\n")
}
