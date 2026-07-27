# Nested CV + stratified repeated CV (Reference §10.13)
# From-scratch base R.
# Run with:  Rscript nested_cv.R

kfold_splits <- function(n, k, seed = 0) {
  set.seed(seed); idx <- sample.int(n)
  folds <- split(idx, cut(seq_along(idx), k, labels = FALSE))
  lapply(seq_len(k), function(i) list(train = unlist(folds[-i]), test = folds[[i]]))
}

nested_cv <- function(X, y, fit_fn, predict_fn, score_fn, hp_grid,
                       k_outer = 5, k_inner = 3, seed = 0) {
  n <- length(y); outer_splits <- kfold_splits(n, k_outer, seed)
  scores <- numeric(k_outer); picked <- vector("list", k_outer)
  for (i in seq_along(outer_splits)) {
    tr <- outer_splits[[i]]$train; te <- outer_splits[[i]]$test
    inner_splits <- kfold_splits(length(tr), k_inner, seed + 1000 + i)
    best_hp <- NA; best_sc <- -Inf
    for (hp in hp_grid) {
      inner_sc <- sapply(inner_splits, function(s) {
        m <- fit_fn(X[tr[s$train], , drop = FALSE], y[tr[s$train]], hp)
        score_fn(y[tr[s$test]], predict_fn(m, X[tr[s$test], , drop = FALSE]))
      })
      if (mean(inner_sc) > best_sc) { best_sc <- mean(inner_sc); best_hp <- hp }
    }
    picked[[i]] <- best_hp
    m <- fit_fn(X[tr, , drop = FALSE], y[tr], best_hp)
    scores[i] <- score_fn(y[te], predict_fn(m, X[te, , drop = FALSE]))
  }
  list(outer_scores = scores, mean_score = mean(scores),
       SE_score = sd(scores) / sqrt(k_outer),
       hp_picked = unlist(picked), k_outer = k_outer, k_inner = k_inner)
}

if (sys.nframe() == 0) {
  set.seed(59); n <- 200
  X <- matrix(rnorm(n * 5), n, 5); beta_true <- c(1, -0.5, 0.3, 0, 0)
  y <- as.vector(X %*% beta_true) + rnorm(n, 0, 0.5)
  ridge_fit <- function(X, y, lam) solve(t(X) %*% X + lam * diag(ncol(X)), t(X) %*% y)
  ridge_pred <- function(b, X) as.vector(X %*% b)
  neg_mse <- function(y, yh) -mean((y - yh)^2)
  cat("=== Nested 5x3 CV for ridge ===\n")
  print(nested_cv(X, y, ridge_fit, ridge_pred, neg_mse, c(0.01, 0.1, 1, 10)))
}
