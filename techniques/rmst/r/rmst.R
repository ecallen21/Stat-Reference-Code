# Restricted Mean Survival Time (Reference §11.29, §11.67)
# Base R + survRM2::rmst2 (authoritative RMST implementation).
# Run with:  Rscript rmst.R

rmst_scratch <- function(times, events, tau) {
  ord <- order(times); t <- times[ord]; e <- events[ord]
  ev <- sort(unique(t[e == 1])); ev <- ev[ev <= tau]
  S <- 1; S_series <- numeric(0); at_risk <- numeric(0); d_events <- numeric(0)
  for (tj in ev) {
    n_j <- sum(t >= tj); d_j <- sum((t == tj) & (e == 1))
    S <- S * (1 - d_j / n_j); S_series <- c(S_series, S)
    at_risk <- c(at_risk, n_j); d_events <- c(d_events, d_j)
  }
  boundaries <- c(0, ev, tau); heights <- c(1, S_series)
  widths <- diff(boundaries); m <- min(length(widths), length(heights))
  rmst <- sum(widths[seq_len(m)] * heights[seq_len(m)])
  # Andersen variance
  T <- sapply(seq_along(ev), function(j) {
    xs <- c(ev[j], if (j + 1 <= length(ev)) ev[(j + 1):length(ev)] else numeric(0), tau)
    ys <- c(S_series[j], if (j + 1 <= length(S_series)) S_series[(j + 1):length(S_series)] else numeric(0))
    sum(diff(xs) * ys[seq_len(length(xs) - 1)])
  })
  var_rmst <- sum(T^2 * d_events / pmax(at_risk * (at_risk - d_events), 1e-12))
  z <- qnorm(0.975); se <- sqrt(max(var_rmst, 0))
  list(RMST = rmst, SE = se,
       CI95_lower = rmst - z * se, CI95_upper = rmst + z * se, tau = tau)
}

if (sys.nframe() == 0) {
  set.seed(37); n <- 200
  group <- sample(0:1, n, replace = TRUE)
  T_event <- rexp(n, rate = ifelse(group == 1, 0.15, 0.08))
  C_censor <- runif(n, 0, 15)
  times <- pmin(T_event, C_censor); events <- as.integer(T_event <= C_censor)
  for (g in 0:1) {
    r <- rmst_scratch(times[group == g], events[group == g], tau = 10)
    cat("group", g, "RMST(10) =", round(r$RMST, 3), "SE =", round(r$SE, 3), "\n")
  }
  if (requireNamespace("survRM2", quietly = TRUE)) {
    cat("\n--- library: survRM2::rmst2 ---\n")
    print(survRM2::rmst2(times, events, group, tau = 10))
  }
}
