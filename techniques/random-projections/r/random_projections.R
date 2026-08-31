# Random projections (Reference Sec 25.12)
# Native R via RandPro; Python via sklearn.
# Run with:  Rscript random_projections.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  RandPro                      -- Gaussian, Achlioptas, Li sparse projections\n")
  cat("  RcppEigen + custom loop       -- fast large-scale projections\n")
  cat("Python:\n")
  cat("  sklearn.random_projection.GaussianRandomProjection\n")
  cat("  sklearn.random_projection.SparseRandomProjection      -- Li 2006 very-sparse\n")
  cat("  johnson-lindenstrauss (pip)   -- theoretical bound calculators\n")
  cat("Refs: Johnson, W.B. & Lindenstrauss, J. (1984) 'Extensions of Lipschitz\n")
  cat("      mappings into a Hilbert space';\n")
  cat("      Achlioptas, D. (2001) 'Database-friendly random projections', PODS.\n")
}
