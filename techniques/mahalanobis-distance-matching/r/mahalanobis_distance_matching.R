# Mahalanobis Distance Matching (Reference Sec 47.326)
# Native R via MatchIt / Matching; Python via causalmatch / from-scratch.
# Run with:  Rscript mahalanobis_distance_matching.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  MatchIt::matchit(distance='mahalanobis')\n")
  cat("  Matching::Match(Weight=2)         -- Mahalanobis by default\n")
  cat("  optmatch::pairmatch (Fullmatch)   -- optimal (min-cost-flow) analogue\n")
  cat("  cobalt::bal.tab                    -- balance diagnostics\n")
  cat("Python:\n")
  cat("  scipy.spatial.distance.cdist(metric='mahalanobis')\n")
  cat("  causalmatch / causalinference (Python packages)\n")
  cat("  From-scratch NN loop (see mahalanobis_distance_matching.py)\n")
  cat("Refs: Rubin, D.B. (1980) 'Bias reduction using Mahalanobis-metric\n")
  cat("      matching', Biometrics 36;  Rosenbaum, P.R. & Rubin, D.B. (1985)\n")
  cat("      'Constructing a control group using multivariate matched sampling\n")
  cat("      methods that incorporate the propensity score', Am Stat 39.\n")
}
