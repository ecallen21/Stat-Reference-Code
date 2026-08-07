# Fine-Gray subdistribution hazards (Reference §11.9)
# R via cmprsk::crr (Gerds-Scheike).
# Run with:  Rscript fine_gray.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 400
  x <- rnorm(n)
  lam1 <- 0.05 * exp(0.6 * x); lam2 <- 0.03 * exp(-0.2 * x)
  T1 <- rexp(n, lam1); T2 <- rexp(n, lam2); C <- rexp(n, 1/30)
  time <- pmin(T1, T2, C)
  event <- rep(0, n)
  event[T1 <= T2 & T1 <= C] <- 1
  event[T2 < T1 & T2 <= C]  <- 2
  if (requireNamespace("cmprsk", quietly = TRUE)) {
    cat("=== cmprsk::crr (Fine-Gray for cause 1) ===\n")
    print(summary(cmprsk::crr(time, event, cov1 = matrix(x, ncol = 1),
                              failcode = 1, cencode = 0)))
  }
}
