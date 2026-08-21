# Geographically Weighted Regression (Reference §23.11)
# R via spgwr::gwr or GWmodel::gwr.basic.
# Run with:  Rscript geographically_weighted_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  spgwr::gwr(formula, data, coords, bandwidth)  -- classic GWR\n")
  cat("  GWmodel::gwr.basic                             -- more kernels + adaptive bandwidth\n")
  cat("  MGWR (Python via mgwr; R via GWmodel with multi-scale)\n")
}
