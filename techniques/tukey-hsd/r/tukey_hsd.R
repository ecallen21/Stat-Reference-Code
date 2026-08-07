# Tukey HSD, Dunnett, Scheffé (Reference §6.9)
# R via stats::TukeyHSD + multcomp::glht for Dunnett/Scheffé.
# Run with:  Rscript tukey_hsd.R

if (sys.nframe() == 0) {
  set.seed(0)
  y <- c(rnorm(20), rnorm(20, 0.5), rnorm(20, 1.5), rnorm(20, 0.2))
  g <- factor(rep(c("A", "B", "C", "D"), each = 20))
  fit <- aov(y ~ g)
  cat("=== TukeyHSD ===\n")
  print(TukeyHSD(fit))
  if (requireNamespace("multcomp", quietly = TRUE)) {
    cat("\n=== multcomp::glht Dunnett vs A ===\n")
    print(summary(multcomp::glht(fit, linfct = multcomp::mcp(g = "Dunnett"))))
  }
}
