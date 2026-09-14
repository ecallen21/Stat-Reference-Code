# FISTA (Reference Sec 47.318)
# Native R via glmnet / celer via reticulate; Python via proxop / from-scratch.
# Run with:  Rscript fista_accelerated_proximal.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  celer via reticulate     -- FISTA + safe screening for LASSO\n")
  cat("  glmnet                   -- coord descent (not FISTA but similar speed)\n")
  cat("  gclm / lassoshooting     -- alternative LASSO solvers\n")
  cat("Python:\n")
  cat("  proxop / pyproximal (general FISTA)\n")
  cat("  sklearn.linear_model.Lasso (coord descent alternative)\n")
  cat("  From-scratch (see fista_accelerated_proximal.py)\n")
  cat("Refs: Beck, A. & Teboulle, M. (2009) 'A fast iterative shrinkage-\n")
  cat("      thresholding algorithm for linear inverse problems',\n")
  cat("      SIAM J Imaging Sci 2.\n")
}
