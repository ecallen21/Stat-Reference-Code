# Expectile regression (Reference Ch 33)
# Native R via expectreg; Python via reticulate.
# Run with:  Rscript expectile_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  expectreg                    -- Sobotka-Kneib expectile regression + splines\n")
  cat("  ExpectReg (fda)              -- FDA-adjacent expectile smoothing\n")
  cat("Python:\n")
  cat("  statsmodels                   -- OLS-based expectile via weight iteration\n")
  cat("  scikit-learn HuberRegressor   -- adjacent (asymmetric-Huber)\n")
  cat("Refs: Newey, W.K. & Powell, J.L. (1987) 'Asymmetric least squares estimation\n")
  cat("      and testing', Econometrica;\n")
  cat("      Bellini, F., Klar, B., Muller, A. & Rosazza Gianin, E. (2014)\n")
  cat("      'Generalized quantiles as risk measures', Insurance: Math & Econ.\n")
}
