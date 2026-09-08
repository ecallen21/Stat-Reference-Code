# Permutation / Kernel SHAP (Reference Sec 47.130)
# Native R via iml / fastshap / DALEX; Python via shap.
# Run with:  Rscript shapley_permutation_explainer.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  iml::Shapley                  -- reference permutation SHAP\n")
  cat("  fastshap::explain             -- efficient MC Shapley + TreeSHAP wrappers\n")
  cat("  DALEX::variable_attribution   -- SHAP + break-down + LIME\n")
  cat("  shapper                       -- SHAP wrapper package\n")
  cat("Python:\n")
  cat("  shap.KernelExplainer / PermutationExplainer / TreeExplainer / DeepExplainer\n")
  cat("  captum.attr.ShapleyValueSampling\n")
  cat("  from-scratch                  -- see shapley_permutation_explainer.py\n")
  cat("Refs: Strumbelj & Kononenko (2010) JMLR 11; Lundberg & Lee (2017)\n")
  cat("      'A unified approach to interpreting model predictions', NeurIPS.\n")
}
