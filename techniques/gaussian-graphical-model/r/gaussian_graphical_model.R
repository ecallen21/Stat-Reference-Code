# Gaussian Graphical Model / Graphical LASSO (Reference Sec 30.8)
# Native R via glasso / huge; Python via sklearn.
# Run with:  Rscript gaussian_graphical_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  glasso                       -- Friedman-Hastie-Tibshirani reference\n")
  cat("  huge                          -- Meinshausen-Buhlmann and MB-based GGMs\n")
  cat("  qgraph                        -- network psychometrics visualisation\n")
  cat("  bootnet                       -- bootstrap-stability for network psychometrics\n")
  cat("Python:\n")
  cat("  sklearn.covariance.GraphicalLasso / GraphicalLassoCV\n")
  cat("  skggm                          -- Bayesian and cross-validated GGM extensions\n")
  cat("  gaussian-graphical-model (pip pkg)\n")
  cat("Refs: Friedman, J., Hastie, T. & Tibshirani, R. (2008) 'Sparse inverse covariance\n")
  cat("      estimation with the graphical lasso', Biostatistics;\n")
  cat("      Meinshausen, N. & Buhlmann, P. (2006) 'High-dimensional graphs and\n")
  cat("      variable selection with the lasso', Annals of Statistics.\n")
}
