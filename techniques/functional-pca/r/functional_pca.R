# Functional PCA (Reference Sec 31.2)
# Native R has excellent FDA support.
# Run with:  Rscript functional_pca.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fda                          -- Ramsay-Silverman reference: pca.fd\n")
  cat("  fda.usc                      -- functional PCA + regression + clustering\n")
  cat("  refund                        -- penalized FPCA (fpca.sc, fpca.face)\n")
  cat("Python:\n")
  cat("  scikit-fda                    -- FPCA + basis smoothing + functional regression\n")
  cat("  fdasrsf                        -- SRSF + phase/amplitude decomposition\n")
  cat("Refs: Ramsay, J.O. & Silverman, B.W. (2005) 'Functional Data Analysis',\n")
  cat("      Springer, Ch. 8; Silverman, B.W. (1996) 'Smoothed functional principal\n")
  cat("      components analysis by choice of norm', Annals of Statistics.\n")
}
