# Parametric AFT survival models (Reference §11.10-§11.15, §11.44, §11.58)
# Base R via survival::survreg + flexsurv for the extra families.
# Run with:  Rscript parametric_survival.R

if (sys.nframe() == 0) {
  set.seed(17); n <- 300
  shape_true <- 1.5; scale_true <- 10
  T_event <- scale_true * (-log(runif(n)))^(1 / shape_true)
  C_censor <- runif(n, 0, 20)
  times <- pmin(T_event, C_censor); events <- as.integer(T_event <= C_censor)
  if (requireNamespace("survival", quietly = TRUE)) {
    cat("=== survival::survreg (Weibull) ===\n")
    print(summary(survival::survreg(survival::Surv(times, events) ~ 1, dist = "weibull")))
    for (d in c("exponential", "weibull", "lognormal", "loglogistic")) {
      f <- survival::survreg(survival::Surv(times, events) ~ 1, dist = d)
      cat(sprintf("%-13s  loglik = %.3f  AIC = %.3f\n", d, f$loglik[2], -2 * f$loglik[2] + 2 * length(f$coefficients)))
    }
  }
}
