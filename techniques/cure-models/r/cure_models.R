# Mixture cure models (Reference §11.22)
# R via smcure::smcure or flexsurvcure::flexsurvcure.
# Run with:  Rscript cure_models.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 500; pi_true <- 0.35
  is_uncured <- runif(n) >= pi_true
  T_ <- rep(Inf, n); T_[is_uncured] <- rweibull(sum(is_uncured), shape = 1.4, scale = 3.0)
  C <- rexp(n, 1 / 5)
  time <- pmin(T_, C); event <- as.integer(T_ <= C)
  df <- data.frame(time = time, event = event)
  if (requireNamespace("flexsurvcure", quietly = TRUE)) {
    cat("=== flexsurvcure::flexsurvcure (Weibull) ===\n")
    fit <- flexsurvcure::flexsurvcure(survival::Surv(time, event) ~ 1,
                                       data = df, dist = "weibull", mixture = TRUE)
    print(fit)
  } else if (requireNamespace("smcure", quietly = TRUE)) {
    cat("=== smcure::smcure ===\n")
    print(smcure::smcure(survival::Surv(time, event) ~ 1, cureform = ~ 1,
                         data = df, model = "ph"))
  }
}
