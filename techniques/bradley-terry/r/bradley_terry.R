# Bradley-Terry pairwise-comparison ranking model (Reference §8.8)
# From-scratch base R via the MM algorithm plus BradleyTerry2::BTm as library
# cross-check.
# Run with:  Rscript bradley_terry.R
#
# Input: `W` -- K x K integer matrix with W[i, j] = wins of i over j (0 diagonal).

fit_bradley_terry <- function(W, item_labels = NULL, max_iter = 1000, tol = 1e-10) {
  W <- as.matrix(W); K <- nrow(W)
  if (is.null(item_labels)) item_labels <- rownames(W)
  if (is.null(item_labels)) item_labels <- paste0("Item", seq_len(K))
  Wi <- rowSums(W); Nij <- W + t(W)
  pi <- rep(1, K)
  for (it in seq_len(max_iter)) {
    pi_new <- numeric(K)
    for (i in seq_len(K)) {
      d <- 0
      for (j in seq_len(K)) if (j != i && Nij[i, j] > 0)
        d <- d + Nij[i, j] / (pi[i] + pi[j])
      pi_new[i] <- if (Wi[i] == 0 || d == 0) 1e-12 else Wi[i] / d
    }
    gmean <- exp(mean(log(pmax(pi_new, 1e-300))))
    pi_new <- pi_new / gmean
    if (max(abs(pi_new - pi) / pmax(pi, 1e-12)) < tol) { pi <- pi_new; break }
    pi <- pi_new
  }
  beta <- log(pi)
  J <- matrix(0, K, K)
  for (i in seq_len(K)) for (j in seq_len(K)) if (i != j && Nij[i, j] > 0) {
    v <- Nij[i, j] * pi[i] * pi[j] / (pi[i] + pi[j])^2
    J[i, i] <- J[i, i] + v; J[i, j] <- J[i, j] - v
  }
  se <- sqrt(pmax(diag(MASS::ginv(J)), 0))
  ord <- order(-pi)
  list(pi = setNames(pi, item_labels),
       beta = setNames(beta, item_labels),
       SE_beta = setNames(se, item_labels),
       ranking = item_labels[ord],
       n_iter = it)
}

if (sys.nframe() == 0) {
  labels <- c("Alice", "Bob", "Carol", "Dan", "Eve")
  W <- matrix(c(0,  4,  6,  8, 10,
                2,  0,  3,  5,  7,
                1,  3,  0,  4,  6,
                1,  2,  3,  0,  5,
                0,  1,  2,  2,  0),
              nrow = 5, byrow = TRUE,
              dimnames = list(labels, labels))
  cat("=== Bradley-Terry MM ===\n"); print(fit_bradley_terry(W, labels))
  if (requireNamespace("BradleyTerry2", quietly = TRUE)) {
    cat("\n--- library: BradleyTerry2::BTm ---\n")
    # convert wins matrix to long form
    df <- expand.grid(winner = labels, loser = labels, stringsAsFactors = FALSE)
    df$freq <- as.vector(W)
    df <- df[df$freq > 0, ]
    print(BradleyTerry2::countsToBinomial(
      xtabs(freq ~ winner + loser, data = df)))
  }
}
