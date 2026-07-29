# Landmark analysis for survival with time-varying exposure (Reference §11.24)
# R via survival package: subset at landmark, then survfit / coxph.
# Run with:  Rscript landmark_analysis.R

if (sys.nframe() == 0) {
  set.seed(9); n <- 500
  T_true <- rexp(n, 1/5); C <- rexp(n, 1/8)
  time <- pmin(T_true, C); event <- as.integer(T_true <= C)
  exposure_time <- rep(Inf, n)
  ever <- runif(n) < 0.5
  exposure_time[ever] <- runif(sum(ever), 0, 4)
  exposure_time[exposure_time > time] <- Inf

  # NAIVE ever/never (biased)
  ever_flag <- is.finite(exposure_time)
  if (requireNamespace("survival", quietly = TRUE)) {
    cat("=== NAIVE ever-vs-never log-rank (biased) ===\n")
    print(survival::survdiff(survival::Surv(time, event) ~ ever_flag))

    cat("\n=== Landmark analysis at t* = 2 ===\n")
    lm_t <- 2.0
    alive <- time > lm_t
    dfl <- data.frame(time = time[alive] - lm_t,
                      event = event[alive],
                      exposed = as.integer(exposure_time[alive] <= lm_t))
    print(survival::survdiff(survival::Surv(time, event) ~ exposed, data = dfl))
  }
}
