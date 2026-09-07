# Overlap weighting -- ATO estimand (Reference Sec 15.31)
# Native R via PSweight; Python via causalml / from-scratch.
# Run with:  Rscript overlap_weighting.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  PSweight::PSweight(..., weight = 'overlap')  -- Li-Morgan-Zaslavsky ATO\n")
  cat("  WeightIt::weightit(..., estimand = 'ATO')    -- overlap weights\n")
  cat("  cobalt::bal.tab                              -- balance diagnostics\n")
  cat("  survey::svyglm                               -- doubly-weighted regression\n")
  cat("Python:\n")
  cat("  causalml.propensity.ElasticNetPropensityModel + custom ATO weights\n")
  cat("  scikit-learn LogisticRegression + hand-rolled ATO formula\n")
  cat("Refs: Li, F., Morgan, K.L. & Zaslavsky, A.M. (2018) 'Balancing covariates\n")
  cat("      via propensity score weighting', JASA 113(521): 390-400.\n")
}
