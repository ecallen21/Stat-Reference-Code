# Variational inference (Reference §14.24, §14.25)
# Base R: mean-field VI on Beta-Binomial and Normal-Normal.
# Production: rstan's ADVI, brms(algorithm = 'meanfield').
# Run with:  Rscript variational_inference.R

if (sys.nframe() == 0) {
  cat("=== CAVI on Beta-Binomial (exact) ===\n")
  a_pr <- 1; b_pr <- 1; y <- 8; n <- 12
  a_q <- a_pr + y; b_q <- b_pr + n - y
  cat(sprintf("  q(theta) = Beta(%g, %g), mean = %.4f\n",
              a_q, b_q, a_q / (a_q + b_q)))

  cat("\n=== Mean-field Gaussian VI on Normal-Normal (analytic) ===\n")
  set.seed(0); y <- rnorm(50, 2.5, 1)
  # Since q Gaussian and target Gaussian, closed-form CAVI IS the exact posterior:
  n <- length(y); prec <- 1 / 100 + n / 1
  v <- 1 / prec; m <- v * (0 / 100 + n * mean(y) / 1)
  cat(sprintf("  q mean = %.4f, sd = %.4f (equals exact posterior)\n", m, sqrt(v)))
}
