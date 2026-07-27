# Cox model residuals + PH test (Reference §11.33, §11.53)
# Base R via survival::coxph + cox.zph for PH test; residuals() for each type.
# Run with:  Rscript cox_diagnostics.R

if (sys.nframe() == 0) {
  set.seed(13); n <- 300
  X <- cbind(x1 = rnorm(n), x2 = rnorm(n))
  lin <- 0.6 * X[, 1] - 0.3 * X[, 2]
  U <- runif(n)
  T_event <- -log(U) / (0.1 * exp(lin))
  C_censor <- runif(n, 0, 20)
  times <- pmin(T_event, C_censor); events <- as.integer(T_event <= C_censor)
  if (requireNamespace("survival", quietly = TRUE)) {
    fit <- survival::coxph(survival::Surv(times, events) ~ X)
    cat("=== Cox fit ===\n"); print(summary(fit))
    cat("\n=== Grambsch-Therneau PH test (cox.zph) ===\n")
    print(survival::cox.zph(fit))
    cat("\n=== Residuals (first 5) ===\n")
    cat("martingale:", head(residuals(fit, type = "martingale"), 5), "\n")
    cat("schoenfeld:", head(residuals(fit, type = "schoenfeld")[, 1], 5), "\n")
    cat("deviance:  ", head(residuals(fit, type = "deviance"), 5), "\n")
  }
}
