# Multilevel 1-1-1 mediation (Reference §12.22)
# Base R via lme4 + RMediation::medci or bmlm for a full multilevel SEM fit.
# Run with:  Rscript multilevel_mediation.R

if (sys.nframe() == 0) {
  cat("For 1-1-1 multilevel mediation in R:\n\n")
  cat("  library(lme4); library(RMediation)\n")
  cat("  # Fit LMM for M and Y with within/between decompositions\n")
  cat("  df$X_b <- ave(df$X, df$id, FUN = mean); df$X_w <- df$X - df$X_b\n")
  cat("  df$M_b <- ave(df$M, df$id, FUN = mean); df$M_w <- df$M - df$M_b\n")
  cat("  fitM <- lmer(M ~ X_w + X_b + (1 | id), data = df)\n")
  cat("  fitY <- lmer(Y ~ M_w + M_b + X_w + X_b + (1 | id), data = df)\n")
  cat("  # Then compute indirect effects a_w*b_w, a_b*b_b and use RMediation for CI.\n\n")
  cat("For a full Bayesian multilevel SEM approach, use the bmlm package.\n")
}
