# LIME -- local interpretable model-agnostic explanations (Reference Sec 47.29)
# Native R via lime; Python via lime (Ribeiro).
# Run with:  Rscript lime_local_explanations.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  lime                  -- Ribeiro et al. reference implementation\n")
  cat("  DALEX + iBreakDown    -- broader explanatory framework, incl. LIME wrapper\n")
  cat("  vip / pdp / iml       -- IML alternatives (permutation importance, PDP, ICE)\n")
  cat("Python:\n")
  cat("  lime (github.com/marcotcr/lime) -- reference implementation\n")
  cat("  shap                             -- SHAP alternative (see shap-values)\n")
  cat("  alibi                            -- Anchor / Counterfactual / IntegratedGradients\n")
  cat("Refs: Ribeiro, M.T., Singh, S. & Guestrin, C. (2016) 'Why should I trust\n")
  cat("      you?: explaining the predictions of any classifier', KDD; Molnar,\n")
  cat("      C. (2022) Interpretable Machine Learning, 2nd ed.\n")
}
