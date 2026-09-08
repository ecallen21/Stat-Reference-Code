# CATE via ICE-curve clustering (Reference Sec 47.51)
# Native R via ICEbox / iml; Python via PyCEbox / dalex.
# Run with:  Rscript cate_clustering_ice.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ICEbox::ice / plot.ice        -- Goldstein et al 2015 reference\n")
  cat("  iml::FeatureEffect             -- ICE / PDP / ALE + clustering helpers\n")
  cat("  DALEX::model_profile           -- profile decomposition; cluster afterwards\n")
  cat("Python:\n")
  cat("  PyCEbox                        -- direct port of ICEbox\n")
  cat("  dalex.model_profile            -- Python DALEX with ICE / PDP\n")
  cat("  sklearn.inspection.PartialDependenceDisplay + KMeans (from-scratch)\n")
  cat("Refs: Goldstein, Kapelner, Bleich & Pitkin (2015) JCGS 24(1); Zhao & Hastie\n")
  cat("      (2021) 'Causal interpretations of black-box models', JBES 39(1).\n")
}
