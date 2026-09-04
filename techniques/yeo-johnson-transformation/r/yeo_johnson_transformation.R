# Yeo-Johnson transformation (Reference Sec 41.2)
# Native R via car::powerTransform (family='yjPower'); Python sklearn + custom.
# Run with:  Rscript yeo_johnson_transformation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  car::powerTransform(family='yjPower') -- MLE lambda with signed support\n")
  cat("  bestNormalize::yeojohnson       -- auto-choose within family\n")
  cat("Python:\n")
  cat("  sklearn.preprocessing.PowerTransformer(method='yeo-johnson')\n")
  cat("  scipy.stats.yeojohnson           -- MLE lambda + transformed data\n")
  cat("Refs: Yeo, I.-K. & Johnson, R.A. (2000) 'A new family of power transformations\n")
  cat("      to improve normality or symmetry', Biometrika.\n")
}
