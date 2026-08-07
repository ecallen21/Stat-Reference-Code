# Mediation analysis (Reference §15.15)
# R via mediation::mediate (Imai-Keele-Tingley causal mediation).
# Run with:  Rscript mediation_analysis.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 300
  T <- rbinom(n, 1, 0.5)
  M <- 1 + 0.8 * T + rnorm(n)
  Y <- 2 + 0.3 * T + 0.5 * M + rnorm(n)
  df <- data.frame(T = T, M = M, Y = Y)
  if (requireNamespace("mediation", quietly = TRUE)) {
    cat("=== mediation::mediate (Imai-Keele-Tingley) ===\n")
    m1 <- lm(M ~ T, data = df); m2 <- lm(Y ~ T + M, data = df)
    med <- mediation::mediate(m1, m2, treat = "T", mediator = "M", sims = 500)
    print(summary(med))
  }
}
