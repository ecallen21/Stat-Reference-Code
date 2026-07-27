# Shared gamma frailty Cox model (Reference §11.26)
# Base R via survival::coxph with frailty() -- the authoritative implementation.
# Run with:  Rscript frailty_models.R

if (sys.nframe() == 0) {
  set.seed(29); n_clusters <- 40; per_c <- 10; n <- n_clusters * per_c
  u <- rgamma(n_clusters, shape = 2, scale = 0.5)
  cluster <- rep(seq_len(n_clusters), each = per_c)
  u_row <- u[cluster]
  X <- matrix(rnorm(n), n, 1); beta_true <- 0.5
  T_event <- -log(runif(n)) / (0.1 * u_row * exp(X %*% beta_true))
  C <- runif(n, 0, 20)
  times <- pmin(T_event, C); events <- as.integer(T_event <= C)

  if (requireNamespace("survival", quietly = TRUE)) {
    cat("=== survival::coxph with frailty(cluster) ===\n")
    print(survival::coxph(survival::Surv(times, events) ~ X +
                            survival::frailty(cluster, distribution = "gamma")))
    cat("\ntrue theta (var(u) / mean(u)^2):", var(u) / mean(u)^2, "\n")
  }
}
