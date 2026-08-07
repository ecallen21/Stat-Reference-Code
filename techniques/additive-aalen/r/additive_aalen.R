# Aalen additive-hazards regression (Reference §11.14)
# R via timereg::aalen (Scheike-Martinussen).
# Run with:  Rscript additive_aalen.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 300
  x <- rnorm(n); lam <- pmax(0.1 + 0.05 * x, 0.001)
  T <- -log(runif(n)) / lam; C <- rexp(n, 1 / 15)
  time <- pmin(T, C); event <- as.integer(T <= C)
  df <- data.frame(time = time, event = event, x = x)
  if (requireNamespace("timereg", quietly = TRUE)) {
    cat("=== timereg::aalen ===\n")
    fit <- timereg::aalen(survival::Surv(time, event) ~ x, data = df)
    print(summary(fit))
  } else if (requireNamespace("survival", quietly = TRUE)) {
    cat("=== survival::aareg (Aalen additive) ===\n")
    fit <- survival::aareg(survival::Surv(time, event) ~ x, data = df)
    print(summary(fit))
  }
}
