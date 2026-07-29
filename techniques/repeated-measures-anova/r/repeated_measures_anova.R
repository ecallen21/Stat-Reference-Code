# Repeated-measures ANOVA + sphericity corrections (Reference §12.1)
# Base R (built-in) or ez::ezANOVA for a cleaner interface.
# Run with:  Rscript repeated_measures_anova.R

rm_anova <- function(Y) {
  Y <- as.matrix(Y); n <- nrow(Y); K <- ncol(Y)
  grand <- mean(Y); subj <- rowMeans(Y); cond <- colMeans(Y)
  SS_total <- sum((Y - grand)^2)
  SS_subj  <- K * sum((subj - grand)^2)
  SS_cond  <- n * sum((cond - grand)^2)
  SS_err   <- SS_total - SS_subj - SS_cond
  df_c <- K - 1; df_e <- (n - 1) * (K - 1)
  MS_c <- SS_cond / df_c; MS_e <- SS_err / df_e
  F_stat <- MS_c / MS_e; p <- pf(F_stat, df_c, df_e, lower.tail = FALSE)
  S <- cov(Y); C <- diag(K) - matrix(1 / K, K, K); A <- C %*% S %*% C
  eps_gg <- (sum(diag(A)))^2 / ((K - 1) * sum(diag(A %*% A)))
  eps_gg <- max(1 / (K - 1), min(1, eps_gg))
  eps_hf <- min(1, (n * (K - 1) * eps_gg - 2) /
                    ((K - 1) * (n - 1 - (K - 1) * eps_gg)))
  list(F = F_stat, df_condition = df_c, df_error = df_e,
       p_uncorrected = p,
       epsilon_GG = eps_gg, epsilon_HF = eps_hf,
       p_GG = pf(F_stat, df_c * eps_gg, df_e * eps_gg, lower.tail = FALSE),
       p_HF = pf(F_stat, df_c * eps_hf, df_e * eps_hf, lower.tail = FALSE),
       partial_eta_squared = SS_cond / (SS_cond + SS_err),
       n_subjects = n, K_conditions = K)
}

if (sys.nframe() == 0) {
  set.seed(3); n <- 30; K <- 4
  subj_eff <- rnorm(n, 0, 0.7); cond_means <- c(0, 0.3, 0.6, 0.4)
  Y <- outer(subj_eff, rep(1, K)) + outer(rep(1, n), cond_means) +
       matrix(rnorm(n * K, 0, 0.5), n, K)
  cat("=== From-scratch RM-ANOVA ===\n"); print(rm_anova(Y))
  cat("\n--- library: stats::aov + built-in RM ---\n")
  long <- data.frame(subject = factor(rep(1:n, K)),
                     condition = factor(rep(1:K, each = n)),
                     y = as.vector(Y))
  print(summary(aov(y ~ condition + Error(subject / condition), data = long)))
}
