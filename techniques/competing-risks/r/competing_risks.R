# Competing-risks analysis (Reference §11.22-§11.25)
# Base R via cmprsk (cuminc + crr) as the authoritative implementations.
# Run with:  Rscript competing_risks.R

if (sys.nframe() == 0) {
  set.seed(19); n <- 200
  T1 <- rexp(n, 0.1); T2 <- rexp(n, 0.05); C <- runif(n, 0, 15)
  T_obs <- pmin(T1, T2, C)
  cause <- ifelse(T1 <= T2, 1, 2)
  cause[pmin(T1, T2) > C] <- 0

  if (requireNamespace("cmprsk", quietly = TRUE)) {
    cat("=== cmprsk::cuminc (Aalen-Johansen CIFs) ===\n")
    ci <- cmprsk::cuminc(T_obs, cause)
    print(ci)

    cat("\n=== cmprsk::crr (Fine-Gray) ===\n")
    X <- matrix(rnorm(n), n, 1)
    print(cmprsk::crr(T_obs, cause, X, failcode = 1))

    cat("\n=== Cause-specific Cox for cause 1 ===\n")
    if (requireNamespace("survival", quietly = TRUE)) {
      print(survival::coxph(survival::Surv(T_obs, as.integer(cause == 1)) ~ X))
    }
  } else cat("cmprsk not installed; skipping.\n")
}
