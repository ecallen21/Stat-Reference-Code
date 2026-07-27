# Nelson-Aalen cumulative hazard estimator (Reference §11.3, §11.65)
# From-scratch base R + survival::survfit(type="fh") as library cross-check.
# Run with:  Rscript nelson_aalen.R

nelson_aalen <- function(times, events) {
  ord <- order(times); t <- times[ord]; e <- events[ord]
  event_times <- sort(unique(t[e == 1]))
  H <- 0; var_H <- 0; H_s <- numeric(0); var_s <- numeric(0)
  at_risk <- numeric(0); d_events <- numeric(0)
  for (tj in event_times) {
    n_j <- sum(t >= tj); d_j <- sum((t == tj) & (e == 1))
    H <- H + d_j / n_j
    var_H <- var_H + d_j / n_j^2
    H_s <- c(H_s, H); var_s <- c(var_s, var_H)
    at_risk <- c(at_risk, n_j); d_events <- c(d_events, d_j)
  }
  z <- qnorm(0.975)
  se_log <- ifelse(H_s > 0, sqrt(var_s) / H_s, 0)
  list(event_times = event_times, n_at_risk = at_risk, d_events = d_events,
       H_hat = H_s, SE_H = sqrt(var_s),
       CI95_lower = H_s * exp(-z * se_log),
       CI95_upper = H_s * exp(z * se_log),
       S_from_H = exp(-H_s),
       n_total = length(times), n_events = sum(events == 1))
}

hazard_rate_smoothed <- function(times, events, grid, bandwidth,
                                  kernel = "epanechnikov") {
  ord <- order(times); t <- times[ord]; e <- events[ord]
  event_times <- t[e == 1]
  n_at_risk <- sapply(event_times, function(tj) sum(t >= tj))
  dH <- 1 / n_at_risk
  K <- switch(kernel,
    "epanechnikov" = function(u) ifelse(abs(u) <= 1, 0.75 * (1 - u^2), 0),
    "gaussian"     = function(u) dnorm(u),
    "uniform"      = function(u) ifelse(abs(u) <= 1, 0.5, 0))
  h_hat <- sapply(grid, function(tg)
    sum(K((tg - event_times) / bandwidth) * dH / bandwidth))
  list(grid = grid, hazard_rate = h_hat, bandwidth = bandwidth, kernel = kernel)
}

if (sys.nframe() == 0) {
  set.seed(5); n <- 100; lambda <- 0.2
  T_event <- rexp(n, lambda); C_censor <- runif(n, 0, 10)
  times <- pmin(T_event, C_censor); events <- as.integer(T_event <= C_censor)
  na <- nelson_aalen(times, events)
  cat("=== Nelson-Aalen ===\n")
  cat("H_hat at first 5:", round(na$H_hat[1:5], 4), "\n")
  cat("theoretical H(t) at t=", na$event_times[20], ":", lambda * na$event_times[20], "\n")
  cat("H_hat there:", na$H_hat[20], "\n")
  hr <- hazard_rate_smoothed(times, events, grid = 1:7, bandwidth = 1.5)
  cat("\n=== Smoothed hazard rate ===\n"); print(round(hr$hazard_rate, 4))
  if (requireNamespace("survival", quietly = TRUE)) {
    cat("\n--- library: survival::survfit(type = 'fh') ---\n")
    s <- survival::survfit(survival::Surv(times, events) ~ 1, type = "fh")
    print(summary(s)$cumhaz[1:5])
  }
}
