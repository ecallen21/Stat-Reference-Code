# Fleiss' kappa: agreement among m >= 3 raters (Reference §8.4)
# From-scratch base R plus irr::kappam.fleiss (library cross-check).
# Run with:  Rscript fleiss_kappa.R
#
# Input: `ratings` -- n x m matrix (rows = items, cols = raters).

fleiss_matrix <- function(ratings, categories = NULL) {
  if (is.null(categories)) categories <- sort(unique(as.vector(ratings)))
  n <- nrow(ratings); K <- length(categories)
  M <- matrix(0L, n, K, dimnames = list(NULL, categories))
  for (i in seq_len(n)) {
    tab <- table(factor(ratings[i, ], levels = categories))
    M[i, ] <- as.integer(tab)
  }
  list(categories = categories, matrix = M)
}

fleiss_kappa <- function(M) {
  n <- nrow(M); K <- ncol(M)
  m <- rowSums(M)
  if (length(unique(m)) != 1) stop("all rows must sum to the same m")
  m <- m[1]
  P_i <- rowSums(M * (M - 1)) / (m * (m - 1))
  P_bar <- mean(P_i)
  p_j <- colSums(M) / (n * m)
  P_e <- sum(p_j^2)
  kappa <- (P_bar - P_e) / (1 - P_e)
  # Fleiss (1971) ASE
  num <- sum(p_j * (1 - p_j))^2 - sum(p_j * (1 - p_j) * (1 - 2 * p_j))
  denom <- sum(p_j * (1 - p_j))^2
  se <- sqrt(2 / (n * m * (m - 1)) * num / denom)
  z <- kappa / se
  # Per-category kappa
  kappa_j <- sapply(seq_len(K), function(j) {
    p_bar_j <- sum(M[, j] * (m - M[, j])) / (n * m * (m - 1))
    dj <- p_j[j] * (1 - p_j[j])
    if (dj > 0) 1 - p_bar_j / dj else NA
  })
  names(kappa_j) <- colnames(M)
  list(kappa = kappa, P_bar = P_bar, P_expected = P_e,
       ASE = se, z = z, p_value = 2 * pnorm(-abs(z)),
       kappa_per_category = kappa_j,
       n_items = n, m_raters = m, K = K)
}

if (sys.nframe() == 0) {
  set.seed(9)
  cats <- c("low", "medium", "high")
  n <- 30; m <- 5
  truth <- sample(cats, n, replace = TRUE)
  ratings <- t(sapply(truth, function(t) sapply(seq_len(m),
      function(.) if (runif(1) < 0.7) t else sample(setdiff(cats, t), 1))))
  fm <- fleiss_matrix(ratings, cats)
  cat("=== Item x category counts (first 5 rows) ===\n"); print(head(fm$matrix, 5))
  cat("\n=== Fleiss' kappa ===\n"); print(fleiss_kappa(fm$matrix))

  if (requireNamespace("irr", quietly = TRUE)) {
    cat("\n--- library: irr::kappam.fleiss ---\n")
    print(irr::kappam.fleiss(ratings))
  } else {
    cat("\n(irr not installed)\n")
  }
}
