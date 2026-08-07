# Robust / cluster-robust standard errors (Reference §5.7, §5.8)
# R via sandwich::vcovHC and sandwich::vcovCL.
# Run with:  Rscript sandwich_robust_se.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 400
  x <- runif(n, -2, 2)
  y <- 1 + 2 * x + (0.5 + abs(x)) * rnorm(n)
  fit <- lm(y ~ x)
  if (requireNamespace("sandwich", quietly = TRUE) &&
      requireNamespace("lmtest", quietly = TRUE)) {
    cat("=== Classical + HC0 + HC3 SEs (heteroscedastic) ===\n")
    print(lmtest::coeftest(fit))
    print(lmtest::coeftest(fit, vcov = sandwich::vcovHC(fit, type = "HC3")))

    # Clustered example
    n_c <- 40; np <- 10; n <- n_c * np
    cl <- rep(1:n_c, each = np)
    u_c <- rnorm(n_c, 0, 2)
    x <- rnorm(n)
    y <- 1 + 2 * x + u_c[cl] + rnorm(n, 0, 0.5)
    fit2 <- lm(y ~ x)
    cat("\n=== Cluster-robust SEs ===\n")
    print(lmtest::coeftest(fit2, vcov = sandwich::vcovCL(fit2, cluster = cl)))
  }
}
