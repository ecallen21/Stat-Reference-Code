# Composite likelihood (Reference Sec 46.13)
# Native R via CompLike / spBayes / geoR; Python via custom.
# Run with:  Rscript composite_likelihood.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  CompRandFld              -- composite likelihood for random fields\n")
  cat("  spBayes                  -- Gaussian process pairwise likelihood\n")
  cat("  geoR / gstat             -- geostatistical composite scoring\n")
  cat("  ordinal + composite      -- pairwise likelihood for ordinal / longitudinal\n")
  cat("  survival / coxme          -- pairwise composite for frailty models\n")
  cat("Python:\n")
  cat("  From-scratch (see accompanying composite_likelihood.py)\n")
  cat("  statsmodels.regression.mixed_linear_model (composite fallback)\n")
  cat("  gpytorch (composite kernels for GP fitting)\n")
  cat("Refs: Lindsay, B.G. (1988) 'Composite likelihood methods', Contemporary\n")
  cat("      Mathematics 80; Varin, C., Reid, N. & Firth, D. (2011) 'An overview\n")
  cat("      of composite likelihood methods', Statistica Sinica 21: 5-42.\n")
}
