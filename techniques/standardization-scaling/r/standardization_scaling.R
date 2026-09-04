# Standardization, centering, scaling (Reference Sec 41.4)
# Native R via base::scale, recipes; Python sklearn + custom.
# Run with:  Rscript standardization_scaling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  base::scale                     -- z-score standardisation\n")
  cat("  recipes::step_normalize / step_center / step_scale / step_range\n")
  cat("Python:\n")
  cat("  sklearn.preprocessing (StandardScaler, MinMaxScaler, RobustScaler,\n")
  cat("     MaxAbsScaler, Normalizer, PowerTransformer)\n")
  cat("Refs: Gelman, A. (2008) 'Scaling regression inputs by dividing by two standard\n")
  cat("      deviations', Stat Med; Enders & Tofighi (2007) 'Centering predictor\n")
  cat("      variables in cross-sectional multilevel models', Psychol Meth.\n")
}
