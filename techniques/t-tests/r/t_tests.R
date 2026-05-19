# One- and two-sample t-tests (Reference §3.4)
# From-scratch base R plus stats::t.test (the idiomatic call).
# Run with:  Rscript t_tests.R
#
# Inputs used below:
#   x, x1, x2  : numeric vectors (the samples; x1, x2 are the two groups)
#   mu0        : null hypothesis value for the mean (one-sample test)
#   equal_var  : TRUE -> Student's pooled t-test; FALSE -> Welch's (default)
#   alternative: "two.sided" / "less" / "greater"
#   conf       : confidence level for the CI on the mean / mean difference
# Sign convention for two-sample tests: mean(x1) - mean(x2).

t_pvalue <- function(t, df, alternative) {
  switch(alternative,
    "two.sided" = 2 * pt(-abs(t), df),
    "greater"   = pt(t, df, lower.tail = FALSE),
    "less"      = pt(t, df),
    stop("alternative must be 'two.sided', 'less', or 'greater'"))
}

# Returns a named numeric vector  c(lower = ..., upper = ...)  so the result
# prints with labels and the caller can read ci["lower"] / ci["upper"].
t_ci <- function(diff, se, df, conf) {
  tcrit <- qt(0.5 + conf / 2, df)
  c(lower = diff - tcrit * se, upper = diff + tcrit * se)
}

one_sample_t_scratch <- function(x, mu0 = 0, alternative = "two.sided", conf = 0.95) {
  n <- length(x); m <- mean(x); s <- sd(x); se <- s / sqrt(n)
  t <- (m - mu0) / se; df <- n - 1
  ci <- t_ci(m, se, df, conf)
  list(mean = m, se = se, t = t, df = df,
       p_value = t_pvalue(t, df, alternative),
       ci_lower = ci[["lower"]], ci_upper = ci[["upper"]])
}

two_sample_t_scratch <- function(x1, x2, equal_var = FALSE,
                                 alternative = "two.sided", conf = 0.95) {
  n1 <- length(x1); n2 <- length(x2)
  m1 <- mean(x1); m2 <- mean(x2); v1 <- var(x1); v2 <- var(x2)
  diff <- m1 - m2
  if (equal_var) {
    sp2 <- ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    se <- sqrt(sp2 * (1/n1 + 1/n2)); df <- n1 + n2 - 2
    d <- diff / sqrt(sp2)
  } else {
    se <- sqrt(v1/n1 + v2/n2)
    df <- (v1/n1 + v2/n2)^2 /
          ((v1/n1)^2 / (n1 - 1) + (v2/n2)^2 / (n2 - 1))
    d <- diff / sqrt((v1 + v2) / 2)
  }
  t <- diff / se; ci <- t_ci(diff, se, df, conf)
  list(mean_diff = diff, se = se, t = t, df = df,
       p_value = t_pvalue(t, df, alternative),
       ci_lower = ci[["lower"]], ci_upper = ci[["upper"]], cohens_d = d,
       method = if (equal_var) "Student" else "Welch")
}

paired_t_scratch <- function(x1, x2, alternative = "two.sided", conf = 0.95) {
  stopifnot(length(x1) == length(x2))
  res <- one_sample_t_scratch(x1 - x2, mu0 = 0, alternative, conf)
  names(res)[1] <- "mean_diff"; res
}

# Library: stats::t.test handles all four cases via its var.equal and paired args
library_demo <- function(x1, x2, mu0 = 100, pre, post) {
  cat("one-sample (mu0 = 100):\n");  print(t.test(x1, mu = mu0))
  cat("\ntwo-sample Welch:\n");       print(t.test(x1, x2))
  cat("\ntwo-sample Student:\n");     print(t.test(x1, x2, var.equal = TRUE))
  cat("\npaired (post vs pre):\n");   print(t.test(post, pre, paired = TRUE))
}

if (sys.nframe() == 0) {
  set.seed(7)
  a <- rnorm(30, 105, 14); b <- rnorm(27, 100, 18)
  pre <- rnorm(20, 50, 8);  post <- pre + rnorm(20, 2, 3)
  cat("=== one-sample t (mu0 = 100) ===\n"); print(one_sample_t_scratch(a, mu0 = 100))
  cat("\n=== two-sample Welch ===\n");        print(two_sample_t_scratch(a, b))
  cat("\n=== two-sample Student ===\n");      print(two_sample_t_scratch(a, b, equal_var = TRUE))
  cat("\n=== paired (post vs pre) ===\n");    print(paired_t_scratch(post, pre))
  cat("\n--- library (stats::t.test) ---\n"); library_demo(a, b, 100, pre, post)
}
