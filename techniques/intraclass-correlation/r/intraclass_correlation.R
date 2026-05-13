# Intraclass correlation coefficient (ICC) (Reference §4.6)
# From-scratch base R plus irr::icc / psych::ICC.
# Run with:  Rscript intraclass_correlation.R
#
# Inputs used below:
#   data  : n x k numeric matrix (subjects x raters)
#   model : "one-way" or "two-way"
#   form  : "absolute" or "consistency" (two-way only)
#   unit  : "single" or "average"

two_way_anova_ms <- function(data) {
  n <- nrow(data); k <- ncol(data); grand <- mean(data)
  row_means <- rowMeans(data); col_means <- colMeans(data)
  ss_r <- k * sum((row_means - grand)^2)
  ss_c <- n * sum((col_means - grand)^2)
  ss_t <- sum((data - grand)^2); ss_e <- ss_t - ss_r - ss_c
  list(MS_R = ss_r / (n - 1), MS_C = ss_c / (k - 1),
       MS_E = ss_e / ((n - 1) * (k - 1)), n = n, k = k)
}

icc_scratch <- function(data, model = "two-way", form = "absolute", unit = "single") {
  n <- nrow(data); k <- ncol(data)
  if (model == "one-way") {
    row_means <- rowMeans(data); grand <- mean(data)
    ss_b <- k * sum((row_means - grand)^2)
    ss_w <- sum((data - row_means)^2)
    ms_b <- ss_b / (n - 1); ms_w <- ss_w / (n * (k - 1))
    val <- if (unit == "single") (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
           else (ms_b - ms_w) / ms_b
    return(list(variant = if (unit == "single") "ICC(1, 1)" else "ICC(1, k)",
                value = val, MS_B = ms_b, MS_W = ms_w, n = n, k = k))
  }
  ms <- two_way_anova_ms(data)
  if (form == "absolute") {
    val <- if (unit == "single")
      (ms$MS_R - ms$MS_E) / (ms$MS_R + (k - 1) * ms$MS_E + k * (ms$MS_C - ms$MS_E) / n)
    else
      (ms$MS_R - ms$MS_E) / (ms$MS_R + (ms$MS_C - ms$MS_E) / n)
    label <- if (unit == "single") "ICC(A, 1)" else "ICC(A, k)"
  } else if (form == "consistency") {
    val <- if (unit == "single") (ms$MS_R - ms$MS_E) / (ms$MS_R + (k - 1) * ms$MS_E)
           else (ms$MS_R - ms$MS_E) / ms$MS_R
    label <- if (unit == "single") "ICC(C, 1)" else "ICC(C, k)"
  } else stop("form must be 'absolute' or 'consistency'")
  list(variant = label, value = val,
       MS_R = ms$MS_R, MS_C = ms$MS_C, MS_E = ms$MS_E, n = n, k = k)
}

# Library: psych::ICC gives all 6 variants in one call; irr::icc gives one
library_demo <- function(data) {
  if (requireNamespace("psych", quietly = TRUE)) {
    cat("psych::ICC:\n"); print(psych::ICC(data))
  } else cat("(install 'psych' for ICC)\n")
}

if (sys.nframe() == 0) {
  set.seed(10); n <- 30; k <- 4
  true_score <- rnorm(n, 50, 10)
  rater_bias <- rnorm(k, 0, 1.5)
  data <- matrix(rep(true_score, k) + rep(rater_bias, each = n) + rnorm(n * k, 0, 3),
                 nrow = n)
  for (cfg in list(c("one-way", "absolute", "single"),
                   c("one-way", "absolute", "average"),
                   c("two-way", "absolute", "single"),
                   c("two-way", "absolute", "average"),
                   c("two-way", "consistency", "single"),
                   c("two-way", "consistency", "average"))) {
    r <- icc_scratch(data, cfg[1], cfg[2], cfg[3])
    cat(sprintf("  %-12s: %+.4f\n", r$variant, r$value))
  }
  cat("\n--- library ---\n"); library_demo(data)
}
