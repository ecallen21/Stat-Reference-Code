# VAR + Cointegration + ECM (Reference §13.12, §13.13, §13.44)
# Base R via vars::VAR + tseries::po.test (Phillips-Ouliaris) + urca::ca.jo (Johansen).
# Run with:  Rscript var_cointegration.R

if (sys.nframe() == 0) {
  set.seed(29); n <- 300
  y2 <- cumsum(rnorm(n))
  y1 <- 2 * y2 + rnorm(n)
  Y <- cbind(y1, y2)
  if (requireNamespace("vars", quietly = TRUE)) {
    cat("=== vars::VAR ===\n"); print(summary(vars::VAR(Y, p = 2)))
  }
  if (requireNamespace("tseries", quietly = TRUE)) {
    cat("\n=== Engle-Granger via tseries::po.test ===\n"); print(tseries::po.test(Y))
  }
  if (requireNamespace("urca", quietly = TRUE)) {
    cat("\n=== Johansen cointegration test ===\n")
    print(summary(urca::ca.jo(Y, type = "trace", K = 2)))
  }
}
