# State-space models + Kalman filter (Reference §13.17, §13.20, §13.55)
# Base R via stats::StructTS + KFAS::KFS + dlm::dlm.
# Run with:  Rscript state_space_kalman.R

if (sys.nframe() == 0) {
  set.seed(41); n <- 100
  x <- cumsum(rnorm(n, 0.1, 0.5))
  y <- ts(x + rnorm(n, 0, 1))
  cat("=== stats::StructTS (level + trend) ===\n")
  print(StructTS(y, type = "trend"))
  if (requireNamespace("KFAS", quietly = TRUE)) {
    cat("\n=== KFAS local-level model ===\n")
    m <- KFAS::SSModel(y ~ SSMtrend(1, Q = list(matrix(NA))), H = matrix(NA))
    print(KFAS::fitSSM(m, inits = c(0, 0))$model)
  }
}
