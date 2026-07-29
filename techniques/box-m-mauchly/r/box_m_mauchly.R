# Box's M and Mauchly's sphericity tests (Reference §9.3, §12.2)
# R via heplots::boxM and stats::mauchly.test (or car::Anova with idata).
# Run with:  Rscript box_m_mauchly.R

if (sys.nframe() == 0) {
  set.seed(0)
  # Box's M -- 3 groups with unequal covariance
  X1 <- MASS::mvrnorm(40, mu = rep(0, 3), Sigma = diag(3))
  X2 <- MASS::mvrnorm(40, mu = rep(0, 3), Sigma = 3 * diag(3))
  X3 <- MASS::mvrnorm(40, mu = rep(0, 3), Sigma = diag(c(1, 5, 1)))
  X <- rbind(X1, X2, X3)
  grp <- factor(rep(c("A", "B", "C"), each = 40))
  if (requireNamespace("heplots", quietly = TRUE)) {
    cat("=== heplots::boxM ===\n")
    print(heplots::boxM(as.data.frame(X), group = grp))
  }
  # Mauchly's sphericity via mauchly.test on an SSD object
  n <- 40; p <- 4; rho <- 0.85
  Sigma <- rho^abs(outer(1:p, 1:p, "-"))
  Y <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  fit <- lm(Y ~ 1)
  cat("\n=== stats::mauchly.test ===\n")
  print(mauchly.test(fit, X = ~1))
}
