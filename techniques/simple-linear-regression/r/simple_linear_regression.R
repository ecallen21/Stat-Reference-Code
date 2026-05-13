# Simple linear regression (Reference §5.1)
# From-scratch base R plus stats::lm. Run with:  Rscript simple_linear_regression.R
#
# Inputs used below:
#   x, y  : paired numeric vectors of equal length (predictor / response)
#   x0    : new x at which to predict
#   conf  : confidence level for the CI / prediction interval
#   kind  : "confidence" -> CI for mean response; "prediction" -> PI for a new obs

fit_scratch <- function(x, y) {
  stopifnot(length(x) == length(y))
  n <- length(x); mx <- mean(x); my <- mean(y)
  Sxx <- sum((x - mx)^2); Sxy <- sum((x - mx) * (y - my)); Syy <- sum((y - my)^2)
  b1 <- Sxy / Sxx; b0 <- my - b1 * mx
  yhat <- b0 + b1 * x; resid <- y - yhat
  rss <- sum(resid^2); df <- n - 2
  sigma2 <- rss / df; sigma <- sqrt(sigma2)
  se_b1 <- sigma / sqrt(Sxx)
  se_b0 <- sigma * sqrt(1/n + mx^2 / Sxx)
  t_b1 <- b1 / se_b1; t_b0 <- b0 / se_b0
  list(n = n, x_bar = mx, y_bar = my, Sxx = Sxx, Sxy = Sxy,
       beta_0 = b0, beta_1 = b1,
       se_beta_0 = se_b0, se_beta_1 = se_b1,
       t_beta_0 = t_b0, t_beta_1 = t_b1,
       p_beta_0 = 2 * pt(-abs(t_b0), df),
       p_beta_1 = 2 * pt(-abs(t_b1), df),
       df_residual = df, rss = rss, sigma_hat = sigma,
       r_squared = 1 - rss / Syy,
       residuals = resid, fitted = yhat)
}

predict_interval_scratch <- function(fit_obj, x0, conf = 0.95, kind = "confidence") {
  yhat <- fit_obj$beta_0 + fit_obj$beta_1 * x0
  base <- 1 / fit_obj$n + (x0 - fit_obj$x_bar)^2 / fit_obj$Sxx
  se <- fit_obj$sigma_hat * sqrt(base + if (kind == "prediction") 1 else 0)
  tc <- qt(0.5 + conf / 2, fit_obj$df_residual)
  list(x0 = x0, yhat = yhat, se = se,
       lower = yhat - tc * se, upper = yhat + tc * se, kind = kind)
}

# Library: stats::lm + summary, predict(... interval = "confidence" / "prediction")
library_demo <- function(x, y) {
  fit <- lm(y ~ x); cat("summary(lm):\n"); print(summary(fit))
  cat("\npredict(... interval = 'confidence') at x = 7:\n")
  print(predict(fit, newdata = data.frame(x = 7), interval = "confidence"))
  cat("predict(... interval = 'prediction') at x = 7:\n")
  print(predict(fit, newdata = data.frame(x = 7), interval = "prediction"))
}

if (sys.nframe() == 0) {
  set.seed(0); n <- 60
  x <- runif(n, 0, 10); y <- 2 + 0.8 * x + rnorm(n, 0, 1.5)
  res <- fit_scratch(x, y)
  cat("=== Fit ===\n")
  for (nm in c("beta_0", "beta_1", "se_beta_0", "se_beta_1",
               "t_beta_0", "t_beta_1", "p_beta_0", "p_beta_1",
               "sigma_hat", "r_squared", "df_residual"))
    cat(sprintf("  %-12s: %s\n", nm, res[[nm]]))
  cat("\n=== 95% CI at x0 = 7 ===\n");          print(predict_interval_scratch(res, 7))
  cat("=== 95% prediction interval at x0 = 7 ===\n")
  print(predict_interval_scratch(res, 7, kind = "prediction"))
  cat("\n--- library ---\n"); library_demo(x, y)
}
