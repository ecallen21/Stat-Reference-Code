# Locally Linear Embedding (Reference Sec 25.15)
# Native R via RDRToolbox / lle; Python via sklearn.
# Run with:  Rscript lle_locally_linear_embedding.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  lle                          -- Kayo Yin LLE + variants (modified LLE, hLLE)\n")
  cat("  RDRToolbox                   -- Isomap, LLE, diffusion maps\n")
  cat("  dimRed                        -- unified manifold-learning wrapper\n")
  cat("Python:\n")
  cat("  sklearn.manifold.LocallyLinearEmbedding   -- standard, modified, hessian, LTSA\n")
  cat("  megaman                        -- large-scale manifold-learning package\n")
  cat("Refs: Roweis, S.T. & Saul, L.K. (2000) 'Nonlinear dimensionality reduction by\n")
  cat("      locally linear embedding', Science 290; Donoho & Grimes (2003) 'Hessian\n")
  cat("      Eigenmaps: Locally linear embedding techniques for high-dim data', PNAS.\n")
}
