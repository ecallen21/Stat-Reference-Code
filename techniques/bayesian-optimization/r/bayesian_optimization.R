# Bayesian optimization (Reference §14.28)
# R via DiceOptim::EGO.nsteps or ParBayesianOptimization::bayesOpt.
# Run with:  Rscript bayesian_optimization.R

if (sys.nframe() == 0) {
  f <- function(x) sin(3 * x) + exp(-(x - 1.5)^2)
  cat("=== Simple 1-D bayesian optimization via grid + GP surrogate ===\n")
  cat("  See DiceOptim::EGO.nsteps for a canonical R implementation.\n")
  if (requireNamespace("ParBayesianOptimization", quietly = TRUE)) {
    r <- ParBayesianOptimization::bayesOpt(
      FUN = function(x) list(Score = f(x)),
      bounds = list(x = c(-3, 4)),
      initPoints = 5, iters.n = 15, plotProgress = FALSE)
    print(r$scoreSummary[which.max(r$scoreSummary$Score), ])
  }
}
