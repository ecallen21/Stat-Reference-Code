# Multiple imputation via MICE (Reference §18.6)
# R via mice::mice + mice::pool.
# Run with:  Rscript multiple_imputation.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 300
  x1 <- rnorm(n); x2 <- 0.5 * x1 + rnorm(n, 0, 0.5)
  y  <- 1 + 2 * x1 - x2 + rnorm(n)
  miss <- runif(n) < plogis(0.6 * x1)
  df <- data.frame(y = y, x1 = x1, x2 = ifelse(miss, NA, x2))
  if (requireNamespace("mice", quietly = TRUE)) {
    cat("=== mice::mice + mice::pool ===\n")
    imp <- mice::mice(df, m = 10, printFlag = FALSE, seed = 1)
    fit <- with(imp, lm(y ~ x1 + x2))
    print(summary(mice::pool(fit)))
  }
}
