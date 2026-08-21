# SHAP values for model explanation (Reference §21.x extra)
# R via shapper, kernelshap, or fastshap.
# Run with:  Rscript shap_values.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  kernelshap::kernelshap(model, X_explain, bg_X)   -- Kernel SHAP (model-agnostic)\n")
  cat("  shapviz::shapviz(shapley_obj)                    -- plots (beeswarm, waterfall)\n")
  cat("  fastshap::explain(model, X, pred_wrapper=predict, nsim=100)  -- Monte-Carlo\n")
  cat("  treeshap::treeshap(unified_model, X)             -- exact TreeSHAP for XGB/LightGBM/RF\n")
  cat("Python: shap.TreeExplainer / shap.KernelExplainer / shap.DeepExplainer\n")
}
