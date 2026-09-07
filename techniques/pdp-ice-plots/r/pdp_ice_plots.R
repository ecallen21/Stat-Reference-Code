# Partial dependence + ICE plots (Reference Sec 47.34)
# Native R via pdp / ICEbox / iml; Python via sklearn.inspection.
# Run with:  Rscript pdp_ice_plots.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  pdp::partial              -- Friedman PDP for many model types\n")
  cat("  ICEbox                    -- Goldstein original ICE implementation\n")
  cat("  iml::FeatureEffect          -- PDP + ICE + ALE + interaction\n")
  cat("  DALEX::model_profile      -- model-profile plots\n")
  cat("Python:\n")
  cat("  sklearn.inspection.partial_dependence / PartialDependenceDisplay\n")
  cat("  PDPbox, pyBreakDown, alibi.explainers.ALE for ALE\n")
  cat("  From-scratch numpy (see pdp_ice_plots.py)\n")
  cat("Refs: Friedman, J.H. (2001) 'Greedy function approximation: a gradient\n")
  cat("      boosting machine', Ann Stat 29(5); Goldstein, A. et al. (2015)\n")
  cat("      'Peeking inside the black box: visualizing statistical learning\n")
  cat("      with plots of individual conditional expectation', JCGS 24(1).\n")
}
