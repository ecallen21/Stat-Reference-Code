# ALE -- accumulated local effects (Reference Sec 47.35)
# Native R via ALEPlot / iml; Python via alibi / from-scratch.
# Run with:  Rscript ale_accumulated_local_effects.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ALEPlot                  -- Apley original implementation\n")
  cat("  iml::FeatureEffect(method = 'ale')  -- ALE via IML framework\n")
  cat("  DALEX::model_profile(type = 'accumulated')\n")
  cat("Python:\n")
  cat("  alibi.explainers.ALE      -- Seldon reference ALE\n")
  cat("  PDPbox includes ALE utilities\n")
  cat("  From-scratch numpy (see ale_accumulated_local_effects.py)\n")
  cat("Refs: Apley, D.W. & Zhu, J. (2020) 'Visualizing the effects of predictor\n")
  cat("      variables in black box supervised learning models', JRSS-B 82(4);\n")
  cat("      Molnar, C. (2022) Interpretable Machine Learning ch 5.3.\n")
}
