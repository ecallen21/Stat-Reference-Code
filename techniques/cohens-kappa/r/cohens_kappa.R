# Cohen's kappa (Reference §8.4)
# From-scratch base R plus irr::kappa2 / psych::cohen.kappa (library cross-check).
# Run with:  Rscript cohens_kappa.R
#
# Inputs:
#   rater1, rater2 : parallel character or integer vectors of category labels

confusion_matrix <- function(rater1, rater2, categories = NULL) {
  if (is.null(categories)) categories <- sort(unique(c(rater1, rater2)))
  tbl <- table(factor(rater1, levels = categories),
               factor(rater2, levels = categories))
  list(categories = categories, matrix = tbl)
}

cohens_kappa <- function(cm) {
  m <- as.matrix(cm); n <- sum(m); K <- nrow(m)
  p_o <- sum(diag(m)) / n
  row <- rowSums(m) / n; col <- colSums(m) / n
  p_e <- sum(row * col)
  kappa <- (p_o - p_e) / (1 - p_e)

  # Fleiss (1969) ASE
  diag_p <- diag(m) / n
  A <- sum(diag_p / (1 - p_e) * (1 - (row + col) * (1 - kappa)))
  off <- 0
  for (i in seq_len(K)) for (j in seq_len(K)) if (i != j) off <- off + m[i, j] / n * (col[i] + row[j])^2
  B <- ((1 - kappa)^2) / (1 - p_e)^2 * off
  C <- (kappa - p_e * (1 - kappa))^2 / (1 - p_e)^2
  var_k <- (A + B - C) / n
  se <- sqrt(max(var_k, 0)); z <- kappa / se
  list(kappa = kappa, p_observed = p_o, p_expected = p_e,
       ASE = se, z = z,
       p_value = 2 * pnorm(-abs(z)),
       CI95_lower = kappa - 1.96 * se, CI95_upper = kappa + 1.96 * se,
       n = n, K = K)
}

pabak <- function(cm) {
  m <- as.matrix(cm); n <- sum(m); K <- nrow(m)
  p_o <- sum(diag(m)) / n
  c(PABAK = (K * p_o - 1) / (K - 1), p_observed = p_o)
}

if (sys.nframe() == 0) {
  set.seed(3)
  cats <- c("low", "medium", "high")
  n <- 200
  r1 <- sample(cats, n, replace = TRUE)
  r2 <- sapply(r1, function(r) if (runif(1) < 0.7) r else sample(setdiff(cats, r), 1))
  cm <- confusion_matrix(r1, r2, cats)
  cat("=== Confusion ===\n"); print(cm$matrix)
  cat("\n=== Cohen's kappa ===\n"); print(cohens_kappa(cm$matrix))
  cat("\n=== PABAK ===\n"); print(pabak(cm$matrix))

  if (requireNamespace("irr", quietly = TRUE)) {
    cat("\n--- library: irr::kappa2 ---\n")
    print(irr::kappa2(data.frame(r1 = r1, r2 = r2)))
  } else {
    cat("\n(irr not installed; skip library cross-check)\n")
  }
  if (requireNamespace("psych", quietly = TRUE)) {
    cat("\n--- library: psych::cohen.kappa ---\n")
    print(psych::cohen.kappa(cbind(r1, r2))$kappa)
  }
}
