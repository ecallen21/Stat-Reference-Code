# Sparse PCA (Reference Sec 25.8)
# Native R via elasticnet / nsprcomp; Python via sklearn.
# Run with:  Rscript sparse_pca.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  elasticnet::spca             -- Zou-Hastie-Tibshirani reference\n")
  cat("  nsprcomp                     -- Sigg-Buhmann Sparse PCA\n")
  cat("  PMA::SPC                     -- Witten-Tibshirani penalised matrix decomp\n")
  cat("Python:\n")
  cat("  sklearn.decomposition.SparsePCA / MiniBatchSparsePCA\n")
  cat("  scikit-optimize sparse-decomp    -- alternating sparse-coding\n")
  cat("Refs: Zou, H., Hastie, T. & Tibshirani, R. (2006) 'Sparse principal component\n")
  cat("      analysis', JCGS; Witten, D.M., Tibshirani, R. & Hastie, T. (2009)\n")
  cat("      'A penalized matrix decomposition, with applications to sparse principal\n")
  cat("      components and canonical correlation analysis', Biostatistics.\n")
}
