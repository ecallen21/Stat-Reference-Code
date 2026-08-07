# Conformal prediction (Reference §10.19)
# R via conformalInference or manual split-conformal.
# Run with:  Rscript conformal_prediction.R

split_conformal <- function(X, y, X_test, alpha = 0.1, calib_frac = 0.5,
                            fit_pred = NULL, seed = 0) {
  set.seed(seed); n <- length(y); perm <- sample(n)
  n_cal <- floor(calib_frac * n)
  cal_idx <- perm[1:n_cal]; tr_idx <- perm[(n_cal + 1):n]
  if (is.null(fit_pred)) {
    fit_pred <- function(Xtr, ytr, Xp) {
      beta <- solve(t(Xtr) %*% Xtr) %*% t(Xtr) %*% ytr
      as.numeric(Xp %*% beta)
    }
  }
  yh <- fit_pred(X[tr_idx, ], y[tr_idx], X[cal_idx, ])
  s <- abs(y[cal_idx] - yh)
  q_lvl <- min(ceiling((n_cal + 1) * (1 - alpha)) / n_cal, 1)
  q <- quantile(s, q_lvl)
  yh_test <- fit_pred(X[tr_idx, ], y[tr_idx], X_test)
  list(prediction = yh_test, lower = yh_test - q, upper = yh_test + q, q = q)
}

if (sys.nframe() == 0) {
  set.seed(0); n <- 500
  X <- cbind(1, matrix(rnorm(n * 3), n, 3))
  y <- X %*% c(1, 2, -1, 0.5) + rnorm(n)
  X_test <- X[1:100, ]; y_test <- y[1:100]
  r <- split_conformal(X[101:n, ], y[101:n], X_test, alpha = 0.1)
  cat(sprintf("empirical 90%% coverage: %.3f  (target 0.90)\n",
              mean(y_test >= r$lower & y_test <= r$upper)))
}
