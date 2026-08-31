# Shape-constrained regression (Reference Sec 33.14)
# Native R has strong support; Python via reticulate.
# Run with:  Rscript shape_constrained_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Iso, isotone                -- PAV isotonic regression\n")
  cat("  scam                         -- shape-constrained additive models\n")
  cat("  cgam                         -- constrained GAM: monotone / convex / concave\n")
  cat("  cobs                         -- constrained B-splines\n")
  cat("Python:\n")
  cat("  sklearn.IsotonicRegression    -- PAV monotone fit\n")
  cat("  cvxpy                          -- monotone / convex / concave QP\n")
  cat("  scipy.optimize.minimize (SLSQP) -- inequality-constrained least squares\n")
  cat("Refs: Barlow, R.E., Bartholomew, D.J., Bremner, J.M. & Brunk, H.D. (1972)\n")
  cat("      'Statistical Inference under Order Restrictions', Wiley;\n")
  cat("      Groeneboom, P. & Jongbloed, G. (2014) 'Nonparametric Estimation under\n")
  cat("      Shape Constraints', Cambridge U.P.\n")
}
