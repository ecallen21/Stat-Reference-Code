# Additive quantile regression (Reference Sec 33.13)
# Native R via quantreg::rqss and qgam; Python via reticulate.
# Run with:  Rscript additive_quantile_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quantreg::rqss              -- Koenker's total-variation smoothing spline QR\n")
  cat("  qgam                         -- Fasiolo et al. calibrated additive QR\n")
  cat("  gamlss.dist / gamlss         -- location-scale-shape (adjacent)\n")
  cat("Python:\n")
  cat("  statsmodels QuantReg + B-spline design matrix\n")
  cat("  pyGAM + custom check-loss smoother (patsy for splines)\n")
  cat("  scikit-learn QuantileRegressor + PolynomialFeatures / SplineTransformer\n")
  cat("Refs: Koenker, R. (2005) 'Quantile Regression', Cambridge U.P., Ch 6;\n")
  cat("      Fasiolo, M., Wood, S.N., Zaffran, M., Nedellec, R. & Goude, Y. (2021)\n")
  cat("      'Fast Calibrated Additive Quantile Regression (qgam)', JASA.\n")
}
