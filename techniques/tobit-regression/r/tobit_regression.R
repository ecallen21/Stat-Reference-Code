# Tobit regression (Reference §5.19)
# R via AER::tobit (survreg with Gaussian).
# Run with:  Rscript tobit_regression.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 400
  x <- rnorm(n); y_star <- 0.5 + 1.0 * x + rnorm(n)
  y <- pmax(y_star, 0)
  if (requireNamespace("AER", quietly = TRUE)) {
    cat("=== AER::tobit (left-censored at 0) ===\n")
    fit <- AER::tobit(y ~ x, left = 0)
    print(summary(fit))
  } else if (requireNamespace("survival", quietly = TRUE)) {
    cat("=== survival::survreg (Gaussian AFT, left-censored) ===\n")
    fit <- survival::survreg(survival::Surv(y, y > 0, type = "left") ~ x,
                             dist = "gaussian")
    print(summary(fit))
  }
}
