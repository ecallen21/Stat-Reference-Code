# SHAP interaction values (Reference Sec 47.40)
# Native R via shapper / kernelshap; Python via shap.
# Run with:  Rscript shap_interactions.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  shapviz + shapviz::sv_interaction  -- SHAP interaction viz on XGB / LGBM\n")
  cat("  kernelshap                           -- Kernel-SHAP wrapper for any model\n")
  cat("  DALEX / iml (integrations)\n")
  cat("Python:\n")
  cat("  shap.TreeExplainer.shap_interaction_values -- fast for tree ensembles\n")
  cat("  shap.KernelExplainer (approximate interactions)\n")
  cat("  From-scratch enumeration (see shap_interactions.py)\n")
  cat("Refs: Lundberg, S.M., Erion, G.G. & Lee, S.-I. (2018) 'Consistent\n")
  cat("      individualized feature attribution for tree ensembles', arXiv:\n")
  cat("      1802.03888; Shapley, L.S. (1953) 'A value for n-person games'.\n")
}
