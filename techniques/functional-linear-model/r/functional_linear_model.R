# Functional linear model (Reference Sec 31.7)
# Native R via fda / refund; Python via scikit-fda.
# Run with:  Rscript functional_linear_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fda::fRegress                -- scalar-on-function + function-on-scalar\n")
  cat("  refund::fosr                  -- Goldsmith et al. function-on-scalar regression\n")
  cat("  refund::pffr                  -- penalised function-on-function regression\n")
  cat("  FDboost                        -- boosted functional regression\n")
  cat("Python:\n")
  cat("  scikit-fda                    -- HistoricalLinearRegression + LinearScalarRegression\n")
  cat("  refund via reticulate         -- Bayesian and penalised variants\n")
  cat("Refs: Ramsay, J.O. & Silverman, B.W. (2005) 'Functional Data Analysis',\n")
  cat("      Springer, Ch. 12-16;\n")
  cat("      Reiss, P.T. et al. (2017) 'Methods for scalar-on-function regression',\n")
  cat("      International Statistical Review.\n")
}
