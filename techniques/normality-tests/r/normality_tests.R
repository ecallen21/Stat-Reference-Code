# Formal tests for normality (Reference §3.19, §3.40)
# From-scratch base R plus idiomatic calls (stats::shapiro.test, nortest::lillie.test,
# nortest::ad.test, tseries::jarque.bera.test).
# Run with:  Rscript normality_tests.R
#
# Inputs used below:
#   x  : numeric vector (the sample)
#   n_mc, rng : Monte-Carlo replications / seed for the from-scratch Lilliefors

central_moment <- function(x, k) { m <- mean(x); mean((x - m)^k) }

jarque_bera_scratch <- function(x) {
  n <- length(x); m2 <- central_moment(x, 2)
  g1 <- central_moment(x, 3) / m2^1.5
  g2 <- central_moment(x, 4) / m2^2 - 3
  jb <- (n / 6) * (g1^2 + g2^2 / 4)
  list(statistic = jb, df = 2,
       p_value = pchisq(jb, df = 2, lower.tail = FALSE),
       skewness = g1, excess_kurtosis = g2)
}

lilliefors_scratch <- function(x, n_mc = 5000, seed = 0) {
  set.seed(seed); n <- length(x)
  mu <- mean(x); sigma <- sd(x); if (sigma == 0) return(NA)
  z_obs <- (x - mu) / sigma
  ks_obs <- max(abs(ecdf(z_obs)(sort(z_obs)) - pnorm(sort(z_obs))))
  sims <- matrix(rnorm(n_mc * n), nrow = n_mc)
  sim_stats <- apply(sims, 1, function(row) {
    z <- (row - mean(row)) / sd(row)
    max(abs(ecdf(z)(sort(z)) - pnorm(sort(z))))
  })
  list(statistic = ks_obs, p_value = mean(sim_stats >= ks_obs),
       note = paste("Monte-Carlo p with", n_mc, "replications"))
}

# Library:
#   stats::shapiro.test           Shapiro-Wilk (3 <= n <= 5000)
#   nortest::lillie.test          Lilliefors (KS with estimated parameters)
#   nortest::ad.test              Anderson-Darling
#   tseries::jarque.bera.test     Jarque-Bera
#   nortest::pearson.test         Pearson chi^2 GOF for normality
library_demo <- function(x) {
  cat("shapiro.test:\n"); print(shapiro.test(x))
  if (requireNamespace("nortest", quietly = TRUE)) {
    cat("\nnortest::lillie.test:\n"); print(nortest::lillie.test(x))
    cat("\nnortest::ad.test:\n");     print(nortest::ad.test(x))
  }
  if (requireNamespace("tseries", quietly = TRUE)) {
    cat("\ntseries::jarque.bera.test:\n"); print(tseries::jarque.bera.test(x))
  }
}

if (sys.nframe() == 0) {
  set.seed(11)
  a <- rnorm(80)
  cat("=== Sample 1: N(0, 1), n = 80 ===\n")
  print(jarque_bera_scratch(a))
  print(lilliefors_scratch(a, n_mc = 2000))
  cat("\n--- library ---\n"); library_demo(a)

  b <- rlnorm(80, 0, 0.7)
  cat("\n=== Sample 2: log-normal, n = 80 ===\n")
  print(jarque_bera_scratch(b))
  print(lilliefors_scratch(b, n_mc = 2000))
  cat("\n--- library ---\n"); library_demo(b)
}
