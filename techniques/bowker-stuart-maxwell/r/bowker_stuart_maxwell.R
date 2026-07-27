# Bowker symmetry + Stuart-Maxwell marginal homogeneity (Reference §8.7, §8.15)
# From-scratch base R plus stats::mcnemar.test (Bowker for K x K) as cross-check.
# Run with:  Rscript bowker_stuart_maxwell.R

build_paired_table <- function(r1, r2, categories = NULL) {
  if (is.null(categories)) categories <- sort(unique(c(r1, r2)))
  as.matrix(table(factor(r1, levels = categories),
                  factor(r2, levels = categories)))
}

bowker_test <- function(tbl) {
  K <- nrow(tbl); stat <- 0
  for (i in seq_len(K - 1)) for (j in (i + 1):K) {
    s <- tbl[i, j] + tbl[j, i]
    if (s > 0) stat <- stat + (tbl[i, j] - tbl[j, i])^2 / s
  }
  df <- K * (K - 1) / 2
  list(chi_square = stat, df = df,
       p_value = pchisq(stat, df, lower.tail = FALSE))
}

stuart_maxwell_test <- function(tbl) {
  K <- nrow(tbl); n <- sum(tbl)
  row <- rowSums(tbl); col <- colSums(tbl)
  if (all(row == col))
    return(list(chi_square = 0, df = K - 1, p_value = 1))
  d <- (row - col)[seq_len(K - 1)]
  V <- matrix(0, K - 1, K - 1)
  for (i in seq_len(K - 1)) {
    V[i, i] <- row[i] + col[i] - 2 * tbl[i, i]
    for (j in seq_len(K - 1)) if (i != j) V[i, j] <- -(tbl[i, j] + tbl[j, i])
  }
  stat <- as.numeric(t(d) %*% solve(V, d))
  list(chi_square = stat, df = K - 1,
       p_value = pchisq(stat, K - 1, lower.tail = FALSE))
}

if (sys.nframe() == 0) {
  set.seed(5)
  cats <- c("improved", "stable", "worsened")
  n <- 300
  before <- sample(cats, n, replace = TRUE)
  idx <- setNames(seq_along(cats), cats)
  trans <- function(c) {
    j <- idx[[c]]; r <- runif(1)
    if (r < 0.6) return(c)
    if (r < 0.9 && j > 1) return(cats[j - 1])
    if (j < length(cats)) return(cats[j + 1])
    c
  }
  after <- sapply(before, trans)
  tbl <- build_paired_table(before, after, cats)
  cat("=== Paired table ===\n"); print(tbl)
  cat("\n=== Bowker symmetry ===\n"); print(bowker_test(tbl))
  cat("\n=== Stuart-Maxwell homogeneity ===\n"); print(stuart_maxwell_test(tbl))
  cat("\n--- library: stats::mcnemar.test (== Bowker for K x K) ---\n")
  print(mcnemar.test(tbl, correct = FALSE))
}
