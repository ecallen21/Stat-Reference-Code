# Panel data: fixed effects + random effects + Hausman (Reference §12.31, §12.32)
# R via plm::plm and plm::phtest.
# Run with:  Rscript fixed_effects_panel.R

if (sys.nframe() == 0) {
  set.seed(0); N <- 100; T <- 5; n <- N * T
  unit <- rep(1:N, each = T)
  time <- rep(1:T, N)
  b_i <- rnorm(N)
  x <- rnorm(n) + 0.5 * b_i[unit]
  z <- rnorm(n)
  y <- 1.2 * x - 0.5 * z + b_i[unit] + rnorm(n, 0, 0.5)
  df <- data.frame(unit = unit, time = time, y = y, x = x, z = z)
  if (requireNamespace("plm", quietly = TRUE)) {
    pdf <- plm::pdata.frame(df, index = c("unit", "time"))
    cat("=== plm::plm within (FE) ===\n")
    print(coef(plm::plm(y ~ x + z, data = pdf, model = "within")))
    cat("\n=== plm::plm random effects ===\n")
    print(coef(plm::plm(y ~ x + z, data = pdf, model = "random")))
    cat("\n=== Hausman test ===\n")
    print(plm::phtest(plm::plm(y ~ x + z, data = pdf, model = "within"),
                      plm::plm(y ~ x + z, data = pdf, model = "random")))
  }
}
