# Coordinate descent lasso (Reference Sec 47.67)
# Native R via glmnet; Python via sklearn / celer / skglm.
# Run with:  Rscript coordinate_descent_lasso.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  glmnet::glmnet         -- Friedman-Hastie-Tibshirani coord descent\n")
  cat("  ncvreg                 -- SCAD / MCP via coord descent\n")
  cat("  biglasso               -- disk-backed glmnet for millions of rows\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.Lasso / LassoCV\n")
  cat("  celer                  -- accelerated coord descent with WS / DS updates\n")
  cat("  skglm                  -- flexible penalties + accelerated CD\n")
  cat("  from-scratch           -- see coordinate_descent_lasso.py\n")
  cat("Refs: Friedman, Hastie & Tibshirani (2010) JSS 33(1); Tibshirani et al\n")
  cat("      (2012) 'Strong rules for discarding predictors', JRSS-B 74(2).\n")
}
