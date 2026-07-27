# K-fold, stratified K-fold, and LOOCV (Reference §10.8, §10.12)
# From-scratch base R + caret::train / boot::cv.glm as library cross-checks.
# Run with:  Rscript cross_validation.R

kfold_indices <- function(n, k, shuffle = TRUE, seed = 0) {
  set.seed(seed); idx <- if (shuffle) sample.int(n) else seq_len(n)
  folds <- split(idx, cut(seq_along(idx), k, labels = FALSE))
  lapply(seq_len(k), function(i) list(test = folds[[i]],
                                       train = unlist(folds[-i])))
}

stratified_kfold_indices <- function(y, k, shuffle = TRUE, seed = 0) {
  set.seed(seed); classes <- unique(y); n <- length(y)
  class_folds <- lapply(classes, function(c) {
    ii <- which(y == c); if (shuffle) ii <- sample(ii)
    split(ii, cut(seq_along(ii), k, labels = FALSE))
  })
  lapply(seq_len(k), function(i) {
    test <- unlist(lapply(class_folds, `[[`, i))
    list(test = test, train = setdiff(seq_len(n), test))
  })
}

cv_score <- function(X, y, fit_fn, predict_fn, score_fn, splits) {
  scores <- sapply(splits, function(s) {
    m <- fit_fn(X[s$train, , drop = FALSE], y[s$train])
    pred <- predict_fn(m, X[s$test, , drop = FALSE])
    score_fn(y[s$test], pred)
  })
  list(fold_scores = scores, mean_score = mean(scores),
       SE_score = sd(scores) / sqrt(length(scores)),
       n_folds = length(scores))
}

if (sys.nframe() == 0) {
  set.seed(41); n <- 200
  X <- cbind(1, rnorm(n), rnorm(n)); y <- 1.5 + 0.8 * X[, 2] - 0.3 * X[, 3] + rnorm(n, 0, 0.5)
  ols_fit <- function(X, y) lm.fit(X, y)$coefficients
  ols_predict <- function(beta, X) as.vector(X %*% beta)
  mse <- function(y_true, y_pred) mean((y_true - y_pred)^2)

  cat("=== 5-fold CV MSE ===\n"); print(cv_score(X, y, ols_fit, ols_predict, mse,
      splits = kfold_indices(n, 5, seed = 0)))
  cat("\n=== 10-fold CV MSE ===\n"); print(cv_score(X, y, ols_fit, ols_predict, mse,
      splits = kfold_indices(n, 10, seed = 0)))
  cat("\n=== LOOCV MSE ===\n"); print(cv_score(X, y, ols_fit, ols_predict, mse,
      splits = kfold_indices(n, n, shuffle = FALSE)))
}
