# IPTW (Reference Sec 15.6)
# Native R via WeightIt / ipw / cobalt; Python causalinference/zepid + custom.
# Run with:  Rscript iptw.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  WeightIt::weightit               -- IPTW with many estimation methods\n")
  cat("  ipw                                -- IPTW for time-varying treatments\n")
  cat("  cobalt                             -- balance diagnostics after IPTW\n")
  cat("  survey::svyglm                     -- design-based inference on weighted data\n")
  cat("Python:\n")
  cat("  causalinference                    -- Rubin-style causal-inference toolbox\n")
  cat("  zepid                               -- IPTW / G-formula / MSM\n")
  cat("  econml (DML / MetaLearners)         -- ML-based causal effects\n")
  cat("  sklearn.linear_model               -- PS model\n")
  cat("Refs: Rosenbaum & Rubin (1983) 'The central role of the propensity score in\n")
  cat("      observational studies for causal effects', Biometrika; Li, Morgan &\n")
  cat("      Zaslavsky (2018) 'Balancing covariates via propensity score weighting',\n")
  cat("      JASA (overlap weights).\n")
}
