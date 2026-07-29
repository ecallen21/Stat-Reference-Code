# Interval-censored survival (Reference §11.20)
# R via icenReg::ic_np (Turnbull) and icenReg::ic_par (parametric).
# Run with:  Rscript interval_censored_survival.R

if (sys.nframe() == 0) {
  set.seed(6); n <- 400
  T_true <- rweibull(n, shape = 1.6, scale = 5)
  visit_grid <- seq(0, 15, by = 1.0)
  L <- numeric(n); R <- rep(Inf, n)
  for (i in 1:n) {
    v <- sort(runif(length(visit_grid) - 1, visit_grid[-length(visit_grid)], visit_grid[-1]))
    before <- v[v < T_true[i]]; after <- v[v >= T_true[i]]
    L[i] <- if (length(before)) tail(before, 1) else 0
    R[i] <- if (length(after)) after[1] else Inf
  }
  df <- data.frame(L = L, R = R)

  if (requireNamespace("icenReg", quietly = TRUE)) {
    cat("=== icenReg::ic_np (Turnbull NPMLE) ===\n")
    print(icenReg::ic_np(cbind(L, R) ~ 0, data = df))

    cat("\n=== icenReg::ic_par (Weibull parametric MLE) ===\n")
    print(icenReg::ic_par(cbind(L, R) ~ 0, data = df, dist = "weibull"))
  } else if (requireNamespace("survival", quietly = TRUE)) {
    cat("=== survival::survfit (Turnbull) ===\n")
    print(survival::survfit(survival::Surv(L, R, type = "interval2") ~ 1, data = df))
  }
}
