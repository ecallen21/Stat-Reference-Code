# Robust location and scale estimators (Reference §1.3 and §1.26)
# From-scratch base-R plus idiomatic package calls (MASS, WRS2, DescTools).
# Run with:  Rscript robust_location_scale.R
#
# Inputs used below:
#   x, y       : numeric vectors (the samples; y is the second group for Yuen's test)
#   proportion : trimming/Winsorizing fraction per tail, in [0, 0.5) (default 0.2)
#   k          : Huber tuning constant in standardized residual units
#                (1.345 -> ~95% efficiency at the normal; smaller -> more robust)
#   tol, max_iter : convergence settings for Huber's IRLS loop
#   scale      : MAD multiplier (1.4826 for the consistent normal-sigma estimate)

mad_scratch <- function(x, scale = 1.4826) {
  med <- median(x); median(abs(x - med)) * scale
}

winsorize_vec <- function(x, proportion = 0.2) {
  stopifnot(proportion >= 0, proportion < 0.5)
  s <- sort(x); n <- length(s); k <- floor(n * proportion)
  if (k > 0) { s[seq_len(k)] <- s[k + 1]; s[(n - k + 1):n] <- s[n - k] }
  s
}

trimmed_mean_scratch <- function(x, proportion = 0.2) {
  s <- sort(x); n <- length(s); k <- floor(n * proportion)
  mean(if (k > 0) s[(k + 1):(n - k)] else s)
}

winsorized_mean_scratch <- function(x, proportion = 0.2) mean(winsorize_vec(x, proportion))

winsorized_var_scratch <- function(x, proportion = 0.2) var(winsorize_vec(x, proportion))

huber_location_scratch <- function(x, k = 1.345, tol = 1e-8, max_iter = 100) {
  mu <- median(x); s <- mad_scratch(x); if (s == 0) s <- 1
  for (i in seq_len(max_iter)) {
    r <- (x - mu) / s
    w <- ifelse(abs(r) <= k, 1, k / abs(r))
    new_mu <- sum(w * x) / sum(w)
    if (abs(new_mu - mu) < tol * max(1, abs(mu))) return(new_mu)
    mu <- new_mu
  }
  mu
}

# Yuen's two-sample trimmed-mean t-test (Welch style)
yuen_trimmed_t <- function(x, y, proportion = 0.2) {
  parts <- function(z) {
    s <- sort(z); n <- length(s); g <- floor(n * proportion); h <- n - 2 * g
    sw <- winsorize_vec(z, proportion); m <- mean(sw)
    ss <- sum((sw - m)^2)
    list(h = h, tm = trimmed_mean_scratch(z, proportion), d = ss / (h * (h - 1)))
  }
  a <- parts(x); b <- parts(y)
  se <- sqrt(a$d + b$d); tstat <- (a$tm - b$tm) / se
  df <- (a$d + b$d)^2 / (a$d^2 / (a$h - 1) + b$d^2 / (b$h - 1))
  list(diff = a$tm - b$tm, se = se, t = tstat, df = df, p_value = 2 * pt(-abs(tstat), df))
}

# Library equivalents:
#   stats::median, stats::mad, mean(x, trim=)
#   MASS::huber (Huber M-estimator of location & scale)
#   WRS2::yuen   (Yuen's trimmed t-test), DescTools::Winsorize
library_versions <- function(x) {
  out <- list(`median (stats)` = median(x),
              `mad scaled (stats)` = mad(x),
              `trimmed mean 20% (base)` = mean(x, trim = 0.2))
  if (requireNamespace("MASS", quietly = TRUE)) {
    h <- MASS::huber(x); out$`huber loc (MASS)` <- h$mu; out$`huber scale (MASS)` <- h$s
  }
  out
}

if (sys.nframe() == 0) {
  data <- c(4, 8, 6, 5, 3, 9, 7, 11, 6, 100)
  cat("data:", data, "\n\n--- from scratch ---\n")
  cat("median             :", median(data), "\n")
  cat("MAD (scaled sigma) :", mad_scratch(data), "\n")
  cat("IQR                :", IQR(data), "\n")
  cat("trimmed mean 20%   :", trimmed_mean_scratch(data, 0.2), "\n")
  cat("winsorized mean 20%:", winsorized_mean_scratch(data, 0.2), "\n")
  cat("winsorized var 20% :", winsorized_var_scratch(data, 0.2), "\n")
  cat("Huber location     :", huber_location_scratch(data), "\n\n--- library ---\n")
  lv <- library_versions(data)
  for (nm in names(lv)) cat(sprintf("%-26s: %s\n", nm, lv[[nm]]))

  cat("\n--- Yuen's trimmed t-test ---\n")
  g1 <- c(10, 12, 11, 9, 13, 10, 11, 50); g2 <- c(14, 15, 13, 16, 14, 15, 14, 13)
  print(yuen_trimmed_t(g1, g2, 0.2))
  if (requireNamespace("WRS2", quietly = TRUE)) print(WRS2::yuen(g1 ~ rep(1:2, c(8, 8))))
}
