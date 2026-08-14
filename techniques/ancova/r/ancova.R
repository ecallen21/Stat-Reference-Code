# Analysis of Covariance (Reference §6.16)
# Base R aov + car::Anova (Type III) + emmeans for adjusted means.
# Run with:  Rscript ancova.R

if (sys.nframe() == 0) {
  set.seed(0); n_per <- 25
  g <- rep(c("A", "B", "C"), each = n_per)
  x <- rnorm(3 * n_per)
  y <- c(1, 2, 3)[match(g, c("A", "B", "C"))] + 0.5 * x + rnorm(3 * n_per)
  df <- data.frame(y = y, g = factor(g), x = x)
  cat("=== ANCOVA with parallel slopes ===\n")
  print(summary(aov(y ~ x + g, data = df)))
  cat("\n=== Interaction (parallel-slopes) test ===\n")
  print(anova(lm(y ~ x + g, data = df), lm(y ~ x * g, data = df)))
  if (requireNamespace("emmeans", quietly = TRUE)) {
    cat("\n=== emmeans adjusted means ===\n")
    print(emmeans::emmeans(lm(y ~ x + g, data = df), specs = "g"))
  }
}
