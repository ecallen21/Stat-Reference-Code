# SINDy (Reference Sec 47.125)
# Native R via sindyr (limited); Python via pysindy.
# Run with:  Rscript sindy_sparse_dynamics.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  sindyr                 -- SINDy for R (limited but exists)\n")
  cat("  glmnet                 -- lasso path for sparse basis coefficients\n")
  cat("Python:\n")
  cat("  pysindy                -- Kutz group reference (polynomial, spline, WSINDy, ...)\n")
  cat("  torchdyn               -- GPU-friendly SINDy variants\n")
  cat("  from-scratch           -- see sindy_sparse_dynamics.py\n")
  cat("Refs: Brunton, Proctor & Kutz (2016) 'Discovering governing equations',\n")
  cat("      PNAS 113(15).\n")
}
