# Recurrent-event Cox models (Reference §11.17-§11.19, §11.41, §11.51)
# Base R via survival::coxph with cluster() and strata().
# Run with:  Rscript recurrent_events.R

if (sys.nframe() == 0) {
  set.seed(23); n_subj <- 100
  X <- matrix(rnorm(n_subj), n_subj, 1); beta_true <- 0.5
  rows_ag <- list(); k_max <- 3
  for (i in seq_len(n_subj)) {
    rate <- 0.3 * exp(X[i, 1] * beta_true); C <- runif(1, 0, 15)
    t <- 0; events <- c()
    while (TRUE) { t <- t + rexp(1, rate); if (t > C) break; events <- c(events, t) }
    prev <- 0
    for (ev in events) {
      rows_ag <- c(rows_ag, list(c(i, prev, ev, 1, X[i, 1]))); prev <- ev
    }
    if (prev < C) rows_ag <- c(rows_ag, list(c(i, prev, C, 0, X[i, 1])))
  }
  df <- do.call(rbind, rows_ag); colnames(df) <- c("id", "start", "stop", "event", "x")
  df <- as.data.frame(df)

  if (requireNamespace("survival", quietly = TRUE)) {
    cat("=== Andersen-Gill (with cluster-robust SE) ===\n")
    print(survival::coxph(survival::Surv(start, stop, event) ~ x + cluster(id), data = df))
  }
}
