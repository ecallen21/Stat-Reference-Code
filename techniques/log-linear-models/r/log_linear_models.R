# Log-linear models for multi-way contingency tables (Reference §8.1, §8.14)
# Base R via stats::glm(family=poisson) and MASS::loglm (Wilkinson-Rogers formulas).
# Run with:  Rscript log_linear_models.R

fit_loglinear_glm <- function(counts, factors_df, formula_rhs) {
  df <- data.frame(y = counts, factors_df)
  form <- as.formula(paste("y ~", formula_rhs))
  m <- glm(form, family = poisson(link = "log"), data = df)
  list(deviance_G2 = m$deviance,
       df = m$df.residual,
       p_value = pchisq(m$deviance, m$df.residual, lower.tail = FALSE),
       expected = fitted(m),
       n_params = length(coef(m)))
}

compare_models <- function(counts, factors_df, small_rhs, large_rhs) {
  s <- fit_loglinear_glm(counts, factors_df, small_rhs)
  l <- fit_loglinear_glm(counts, factors_df, large_rhs)
  list(small_G2 = s$deviance_G2, large_G2 = l$deviance_G2,
       delta_G2 = s$deviance_G2 - l$deviance_G2,
       delta_df = s$df - l$df,
       p_value = pchisq(s$deviance_G2 - l$deviance_G2,
                        s$df - l$df, lower.tail = FALSE))
}

fit_agreement_models <- function(square_counts) {
  M <- as.matrix(square_counts); K <- nrow(M)
  rows <- rep(seq_len(K), each = K); cols <- rep(seq_len(K), K)
  y <- as.vector(t(M))                # row-major
  df <- data.frame(y = y,
                   R = factor(rows, levels = seq_len(K)),
                   C = factor(cols, levels = seq_len(K)))
  df$Diag <- factor(ifelse(rows == cols, rows, 0))
  # symmetric pair indicator: which unordered {i, j} pair each cell belongs to
  pair_id <- pmin(rows, cols) * (K + 1) + pmax(rows, cols)
  df$Pair <- factor(pair_id)
  fit <- function(rhs) {
    m <- glm(as.formula(paste("y ~", rhs)), family = poisson, data = df)
    list(G2 = m$deviance, df = m$df.residual,
         p_value = pchisq(m$deviance, m$df.residual, lower.tail = FALSE),
         n_params = length(coef(m)))
  }
  list(independence = fit("R + C"),
       quasi_independence = fit("R + C + Diag"),
       quasi_symmetry = fit("R + C + Pair"))
}

if (sys.nframe() == 0) {
  # 2x2x2 example
  counts <- c(50, 60, 30, 40, 20, 45, 25, 35)
  grid <- expand.grid(A = factor(0:1), B = factor(0:1), C = factor(0:1))
  cat("=== [A][B][C] mutual independence ===\n")
  print(fit_loglinear_glm(counts, grid, "A + B + C"))
  cat("\n=== [AB][AC][BC] no 3-way ===\n")
  print(fit_loglinear_glm(counts, grid, "A + B + C + A:B + A:C + B:C"))
  cat("\n=== LR test: indep vs all-2-way ===\n")
  print(compare_models(counts, grid, "A + B + C", "A + B + C + A:B + A:C + B:C"))

  if (requireNamespace("MASS", quietly = TRUE)) {
    cat("\n--- library: MASS::loglm ---\n")
    arr <- array(counts, dim = c(2, 2, 2))
    print(MASS::loglm(~ 1 + 2 + 3, data = arr))                # mutual indep
    print(MASS::loglm(~ 1*2 + 1*3 + 2*3, data = arr))          # all 2-way
  }

  # 4x4 agreement example
  square <- matrix(c(50, 8, 1, 0,
                     7, 30, 6, 1,
                     1, 5, 25, 4,
                     0, 1, 3, 20), nrow = 4, byrow = TRUE)
  cat("\n=== Agreement models (4x4) ===\n")
  print(fit_agreement_models(square))
}
