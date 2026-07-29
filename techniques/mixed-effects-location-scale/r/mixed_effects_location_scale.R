# Mixed-Effects Location-Scale (MELS) model (Reference §12.21)
# R via mixregls (Hedeker & Nordgren 2013) or nlme with varIdent.
# Run with:  Rscript mixed_effects_location_scale.R

if (sys.nframe() == 0) {
  cat("For a full joint MELS fit in R:\n\n")
  cat("  library(mixregls)\n")
  cat("  fit <- mixregls_fit(y ~ x, id = 'subject', data = df,\n")
  cat("                       scale_model = ~ 1)\n\n")
  cat("Alternative: nlme::lme with a varIdent variance function to allow\n")
  cat("per-subject residual variance heterogeneity (less flexible but faster).\n")
}
