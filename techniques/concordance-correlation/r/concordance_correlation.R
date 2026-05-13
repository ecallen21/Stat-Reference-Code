# Lin's concordance correlation coefficient (Reference §4.9)
# From-scratch base R plus DescTools::CCC (also epiR::epi.ccc).
# Run with:  Rscript concordance_correlation.R
#
# Inputs used below:
#   x, y : two paired numeric vectors of equal length (same measurement, two devices)
#   conf : confidence level for the Fisher-z CI on rho_c

lins_ccc_scratch <- function(x, y, conf = 0.95) {
  stopifnot(length(x) == length(y))
  n <- length(x); mx <- mean(x); my <- mean(y)
  vx <- sum((x - mx)^2) / n; vy <- sum((y - my)^2) / n
  cov <- sum((x - mx) * (y - my)) / n
  rho_c <- (2 * cov) / (vx + vy + (mx - my)^2)
  sx <- sqrt(vx); sy <- sqrt(vy)
  pearson <- cov / (sx * sy)
  v <- sx / sy; u <- (mx - my) / sqrt(sx * sy)
  C_b <- 2 / (v + 1/v + u^2)
  z <- atanh(rho_c); se <- 1 / sqrt(n - 3); zc <- qnorm(0.5 + conf / 2)
  list(ccc = rho_c, pearson_r_precision = pearson, C_b_accuracy = C_b,
       scale_shift_v = v, location_shift_u = u,
       ci_lower = tanh(z - zc * se), ci_upper = tanh(z + zc * se), n = n)
}

# Library: DescTools::CCC (also epiR::epi.ccc)
library_demo <- function(x, y) {
  if (requireNamespace("DescTools", quietly = TRUE)) {
    cat("DescTools::CCC:\n"); print(DescTools::CCC(x, y))
  } else cat("(install 'DescTools' for CCC)\n")
}

if (sys.nframe() == 0) {
  set.seed(12)
  cat("=== Strong correlation but bias of +5 ===\n")
  x <- rnorm(60, 50, 10); y <- x + 5 + rnorm(60)
  print(lins_ccc_scratch(x, y))
  cat("Pearson for reference:", cor(x, y), "\n")
  cat("\n=== Perfect agreement (y = x) ===\n"); print(lins_ccc_scratch(x, x))
  cat("\n=== Same correlation, scale shift (y = 2x + noise) ===\n")
  print(lins_ccc_scratch(x, 2 * x + rnorm(60)))
  cat("\n--- library ---\n"); library_demo(x, x + 5 + rnorm(60))
}
