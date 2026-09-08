# Feature-attribution stability (Reference Sec 47.52)
# Native R via iml + custom; Python via shap / captum / alibi.
# Run with:  Rscript attribution_stability.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  iml::FeatureImp        -- permutation importance; bootstrap in a loop\n")
  cat("  DALEX::model_parts     -- model-agnostic attribution + repeat= arg\n")
  cat("  fastshap               -- SHAP values with bootstrap wrappers\n")
  cat("Python:\n")
  cat("  shap                   -- Tree / Kernel / Deep SHAP + repeatability tests\n")
  cat("  captum                 -- gradient-based attributions for PyTorch\n")
  cat("  alibi.explainers       -- IntegratedGradients, KernelShap, Anchors\n")
  cat("  from-scratch           -- see attribution_stability.py\n")
  cat("Refs: Yeh et al (2019) 'On the (in)fidelity and sensitivity of explanations',\n")
  cat("      NeurIPS; Alvarez-Melis & Jaakkola (2018) 'On the robustness of\n")
  cat("      interpretability methods', WHI ICML.\n")
}
