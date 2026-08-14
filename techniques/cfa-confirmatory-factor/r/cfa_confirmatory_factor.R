# Confirmatory Factor Analysis (Reference §19.5)
# R via lavaan::cfa (Rosseel).
# Run with:  Rscript cfa_confirmatory_factor.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 500
  F <- MASS::mvrnorm(n, c(0, 0), matrix(c(1, 0.4, 0.4, 1), 2))
  lam <- matrix(c(0.8, 0.7, 0.75, 0, 0, 0,
                  0, 0, 0, 0.85, 0.7, 0.65), 6, 2)
  err_sd <- c(0.6, 0.7, 0.65, 0.55, 0.7, 0.75)
  X <- F %*% t(lam) + matrix(rnorm(n * 6), n, 6) * matrix(err_sd, n, 6, byrow = TRUE)
  colnames(X) <- paste0("x", 1:6)
  df <- as.data.frame(X)
  if (requireNamespace("lavaan", quietly = TRUE)) {
    model <- '
      f1 =~ x1 + x2 + x3
      f2 =~ x4 + x5 + x6
      f1 ~~ f2
    '
    cat("=== lavaan::cfa ===\n")
    fit <- lavaan::cfa(model, data = df)
    print(lavaan::fitMeasures(fit, c("chisq", "df", "cfi", "rmsea", "srmr")))
  }
}
