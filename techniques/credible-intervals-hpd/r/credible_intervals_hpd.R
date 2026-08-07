# Credible intervals and ROPE (Reference §14.9, §14.23)
# Base R: equal-tail interval + HPD interval (assumes unimodal posterior).
# Production: HDInterval::hdi, bayestestR::hdi, bayestestR::rope.
# Run with:  Rscript credible_intervals_hpd.R

hpd <- function(x, alpha = 0.05) {
  x <- sort(x); n <- length(x); k <- floor((1 - alpha) * n)
  widths <- x[(k + 1):n] - x[1:(n - k)]
  i <- which.min(widths)
  c(x[i], x[i + k])
}

rope_decision <- function(x, rope, cred = 0.95) {
  h <- hpd(x, alpha = 1 - cred)
  if (h[1] > rope[2] || h[2] < rope[1]) "reject null"
  else if (h[1] >= rope[1] && h[2] <= rope[2]) "accept null"
  else "withhold judgment"
}

if (sys.nframe() == 0) {
  set.seed(0)
  cat("=== Skewed lognormal posterior ===\n")
  x <- rlnorm(5000, 1, 0.7)
  cat(sprintf("  ETI: (%.3f, %.3f)  width = %.3f\n",
              quantile(x, 0.025), quantile(x, 0.975),
              diff(quantile(x, c(0.025, 0.975)))))
  h <- hpd(x); cat(sprintf("  HPD: (%.3f, %.3f)  width = %.3f\n",
                            h[1], h[2], diff(h)))

  cat("\n=== ROPE decision on three effect sizes ===\n")
  for (nm in names(list(pos = rnorm(5000, 0.30, 0.05),
                        null = rnorm(5000, 0.01, 0.03),
                        amb  = rnorm(5000, 0.05, 0.08)))) {
    x <- switch(nm,
                pos  = rnorm(5000, 0.30, 0.05),
                null = rnorm(5000, 0.01, 0.03),
                amb  = rnorm(5000, 0.05, 0.08))
    cat(sprintf("  %-4s: HDI = (%.3f, %.3f)  decision = %s\n",
                nm, hpd(x)[1], hpd(x)[2],
                rope_decision(x, c(-0.05, 0.05))))
  }
}
