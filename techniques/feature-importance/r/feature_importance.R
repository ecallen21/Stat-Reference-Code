# Feature importance (Reference §26.16)
# R via iml (Molnar) or DALEX (Biecek).
# Run with:  Rscript feature_importance.R

if (sys.nframe() == 0) {
  cat("Recommended R packages:\n")
  cat("  iml::FeatureImp    -- permutation importance\n")
  cat("  iml::FeatureEffect -- PDP and ICE\n")
  cat("  DALEX::feature_importance / model_profile\n")
  cat("  vip / pdp for tree-based models\n")
}
