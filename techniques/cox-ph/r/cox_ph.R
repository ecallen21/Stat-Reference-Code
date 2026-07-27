# Cox proportional-hazards model (Reference §11.8, §11.16, §11.42, §11.54, §11.63, §11.64, §11.66)
# Base R via survival::coxph as library cross-check.
# Run with:  Rscript cox_ph.R
#
# The from-scratch implementation lives in the Python file; base R already
# ships an authoritative Cox implementation in the survival package, so the R
# version here uses it directly.

if (sys.nframe() == 0) {
  set.seed(9); n <- 200
  X <- cbind(x1 = rnorm(n), x2 = rnorm(n))
  beta_true <- c(0.7, -0.4)
  lin <- as.vector(X %*% beta_true)
  U <- runif(n)
  T_event <- -log(U) / (0.1 * exp(lin))
  C_censor <- runif(n, 0, 15)
  times <- pmin(T_event, C_censor); events <- as.integer(T_event <= C_censor)

  if (requireNamespace("survival", quietly = TRUE)) {
    cat("=== survival::coxph (Efron ties, default) ===\n")
    fit <- survival::coxph(survival::Surv(times, events) ~ X)
    print(summary(fit))
    cat("\n=== EPV rule check ===\n")
    cat("events =", sum(events == 1), ", p =", ncol(X),
        ", EPV =", sum(events == 1) / ncol(X), "\n")
  }
}
