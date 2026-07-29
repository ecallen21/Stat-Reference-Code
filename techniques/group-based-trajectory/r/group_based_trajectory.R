# Group-Based Trajectory Modeling (Reference §12.5, §12.6, §12.7)
# Base R via crimCV / lcmm packages.
# Run with:  Rscript group_based_trajectory.R

if (sys.nframe() == 0) {
  cat("For group-based / latent-class trajectory models in R:\n\n")
  cat("  library(lcmm)         # unified latent-class LMM (LCGA, GBTM, GMM)\n")
  cat("  fit <- lcmm::hlme(y ~ time + I(time^2), subject = 'id',\n")
  cat("                     ng = 3, data = df,\n")
  cat("                     mixture = ~ time + I(time^2))\n")
  cat("  summary(fit)\n\n")
  cat("  # Or crimCV::crimCV() for count-outcome GBTM (criminology origin).\n\n")
  cat("For growth mixture models (GMM = GBTM + within-class random effects):\n")
  cat("  Add nwg = TRUE to lcmm::hlme() for group-specific random-effect variances.\n")
}
