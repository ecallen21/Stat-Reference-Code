# Latent profile analysis (Reference Sec 36.2)
# Native R via mclust / tidyLPA; Python sklearn.mixture.
# Run with:  Rscript latent_profile_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mclust::Mclust                   -- Gaussian mixture with BIC-based K selection\n")
  cat("  tidyLPA                           -- tidyverse-friendly LPA wrapper\n")
  cat("  MplusAutomation                   -- LPA / LCA via Mplus from R\n")
  cat("Python:\n")
  cat("  sklearn.mixture.GaussianMixture   -- diagonal / full / spherical covariance\n")
  cat("  bnpy / pomegranate                 -- alternative mixture models\n")
  cat("Refs: Gibson (1959) 'Three multivariate models: factor analysis, latent structure\n")
  cat("      analysis, and latent profile analysis', Psychometrika; Vermunt & Magidson\n")
  cat("      (2002) 'Latent class cluster analysis' in Applied Latent Class Analysis.\n")
}
