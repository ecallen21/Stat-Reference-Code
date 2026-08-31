# Robust PCA (Reference Sec 25.13)
# Native R via rpca; Python via reticulate.
# Run with:  Rscript robust_pca.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rpca                         -- Candes 2011 PCP reference\n")
  cat("  pcaPP                        -- projection-pursuit robust PCA\n")
  cat("  robustbase                   -- robust covariance + robust PCA\n")
  cat("Python:\n")
  cat("  fbpca                        -- fast randomised PCA (adjacent)\n")
  cat("  splitpca (community)         -- Principal Component Pursuit\n")
  cat("  ristretto                    -- randomised low-rank + sparse solvers\n")
  cat("Refs: Candes, E.J., Li, X., Ma, Y. & Wright, J. (2011) 'Robust principal\n")
  cat("      component analysis?', Journal of the ACM;\n")
  cat("      Lin, Z., Chen, M. & Ma, Y. (2010) 'The augmented Lagrange multiplier\n")
  cat("      method for exact recovery of corrupted low-rank matrices', arXiv:1009.5055.\n")
}
