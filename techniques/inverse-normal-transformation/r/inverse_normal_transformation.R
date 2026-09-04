# Rank-based inverse normal transformation (Reference Sec 41.3)
# Native R via RNOmni; Python custom + scipy.
# Run with:  Rscript inverse_normal_transformation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  RNOmni::RankNorm                -- Blom / Tukey / vdW variants\n")
  cat("  bestNormalize::orderNorm        -- Ordered quantile normalization\n")
  cat("Python:\n")
  cat("  scipy.stats.rankdata + norm.ppf\n")
  cat("  sklearn.preprocessing.QuantileTransformer(output_distribution='normal')\n")
  cat("Refs: Beasley, Erickson & Allison (2009) 'Rank-based inverse normal\n")
  cat("      transformations are increasingly used, but are they merited?', Behav Genet;\n")
  cat("      Blom, G. (1958) Statistical Estimates and Transformed Beta-Variables, Wiley.\n")
}
