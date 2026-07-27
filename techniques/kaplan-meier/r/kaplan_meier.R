# Kaplan-Meier survival estimator (Reference §11.2)
# From-scratch base R + survival::survfit as library cross-check.
# Run with:  Rscript kaplan_meier.R

kaplan_meier <- function(times, events, weights = NULL) {
  n <- length(times)
  if (is.null(weights)) weights <- rep(1, n)
  ord <- order(times); t <- times[ord]; e <- events[ord]; w <- weights[ord]
  event_times <- sort(unique(t[e == 1]))
  cum <- 1; S <- numeric(0); at_risk <- numeric(0); d_events <- numeric(0); var_S <- numeric(0)
  green_sum <- 0
  for (tj in event_times) {
    n_j <- sum(w[t >= tj])
    d_j <- sum(w[(t == tj) & (e == 1)])
    if (n_j <= 0) next
    cum <- cum * (1 - d_j / n_j)
    denom <- n_j * (n_j - d_j)
    if (denom > 0) green_sum <- green_sum + d_j / denom
    var_j <- cum^2 * green_sum
    S <- c(S, cum); at_risk <- c(at_risk, n_j); d_events <- c(d_events, d_j); var_S <- c(var_S, var_j)
  }
  z <- qnorm(0.975)
  ci_lo <- ci_hi <- numeric(length(S))
  for (i in seq_along(S)) {
    s <- S[i]; v <- var_S[i]
    if (s > 0 && s < 1 && v > 0) {
      g <- log(-log(s)); se_g <- sqrt(v) / (s * abs(log(s)))
      ci_hi[i] <- exp(-exp(g - z * se_g))
      ci_lo[i] <- exp(-exp(g + z * se_g))
    } else { ci_lo[i] <- ci_hi[i] <- s }
  }
  list(event_times = event_times, n_at_risk = at_risk, d_events = d_events,
       S_hat = S, SE_S = sqrt(var_S), CI95_lower = ci_lo, CI95_upper = ci_hi,
       n_total = n, n_events = sum(events == 1))
}

median_survival <- function(km) {
  S <- km$S_hat; t <- km$event_times; v <- km$SE_S^2
  below <- which(S <= 0.5)
  med <- if (length(below)) t[below[1]] else Inf
  z <- qnorm(0.975)
  inside <- t[(S - 0.5)^2 <= z^2 * v & v > 0]
  list(median = med, CI95_lower = if (length(inside)) min(inside) else NA,
       CI95_upper = if (length(inside)) max(inside) else NA)
}

if (sys.nframe() == 0) {
  set.seed(3); n <- 100; lambda <- 0.15
  T_event <- rexp(n, rate = lambda); C_censor <- runif(n, 0, 12)
  times <- pmin(T_event, C_censor); events <- as.integer(T_event <= C_censor)
  km <- kaplan_meier(times, events)
  cat("=== Kaplan-Meier (n =", km$n_total, ", events =", km$n_events, ") ===\n")
  cat("S_hat at first 5 event times:\n"); print(round(km$S_hat[1:5], 4))
  ms <- median_survival(km)
  cat("\n=== Median survival ===\n")
  cat("median =", round(ms$median, 4), "  BC 95% CI: [", ms$CI95_lower, ",", ms$CI95_upper, "]\n")
  cat("theoretical median =", round(log(2) / lambda, 4), "\n")
  if (requireNamespace("survival", quietly = TRUE)) {
    cat("\n--- library: survival::survfit ---\n")
    print(survival::survfit(survival::Surv(times, events) ~ 1))
  }
}
