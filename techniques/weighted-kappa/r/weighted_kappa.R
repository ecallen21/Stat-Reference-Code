# Weighted kappa for ordinal agreement (Reference §8.4)
# From-scratch base R plus irr::kappa2 with weights (library cross-check).
# Run with:  Rscript weighted_kappa.R

build_weight_matrix <- function(K, scheme = "quadratic") {
  D <- abs(outer(seq_len(K), seq_len(K), "-"))
  if (scheme == "linear")    return(D / (K - 1))
  if (scheme == "quadratic") return(D^2 / (K - 1)^2)
  stop("scheme must be 'linear' or 'quadratic'")
}

confusion_matrix <- function(r1, r2, categories) {
  tbl <- table(factor(r1, levels = categories),
               factor(r2, levels = categories))
  as.matrix(tbl)
}

kappa_w_point <- function(p, W) {
  row <- rowSums(p); col <- colSums(p)
  p_o <- sum(W * p)
  p_e <- sum(W * outer(row, col))
  if (p_e == 0) return(list(kappa = 1 - p_o, p_o = p_o, p_e = p_e))
  list(kappa = 1 - p_o / p_e, p_o = p_o, p_e = p_e)
}

weighted_kappa <- function(cm, scheme = "quadratic", n_boot = 2000, seed = 0) {
  m <- as.matrix(cm)
  K <- nrow(m); n <- sum(m)
  p <- m / n
  W <- build_weight_matrix(K, scheme)
  est <- kappa_w_point(p, W)

  set.seed(seed)
  flat <- as.vector(p)
  boot <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    picks <- sample.int(K * K, size = n, replace = TRUE, prob = flat)
    counts <- matrix(tabulate(picks, nbins = K * K), nrow = K)
    boot[b] <- kappa_w_point(counts / n, W)$kappa
  }
  se <- sd(boot); ci <- quantile(boot, c(0.025, 0.975))
  list(kappa_weighted = est$kappa, scheme = scheme,
       weighted_p_o = est$p_o, weighted_p_e = est$p_e,
       SE_bootstrap = se,
       CI95_lower = unname(ci[1]),
       CI95_upper = unname(ci[2]),
       n_boot = n_boot, K = K, n = n)
}

if (sys.nframe() == 0) {
  set.seed(4)
  cats <- c("mild", "moderate", "severe", "critical")
  n <- 200
  r1 <- sample(cats, n, replace = TRUE)
  idx <- setNames(seq_along(cats), cats)
  perturb <- function(c) {
    j <- idx[[c]]; r <- runif(1)
    if (r < 0.6) return(c)
    if (r < 0.9) sh <- sample(c(-1, 1), 1) else sh <- sample(c(-2, 2), 1)
    cats[max(1, min(length(cats), j + sh))]
  }
  r2 <- sapply(r1, perturb)
  cm <- confusion_matrix(r1, r2, cats)
  cat("=== Confusion ===\n"); print(cm)
  for (sc in c("linear", "quadratic")) {
    cat("\n===", sc, "===\n"); print(weighted_kappa(cm, sc, n_boot = 1000))
  }
  if (requireNamespace("irr", quietly = TRUE)) {
    cat("\n--- library: irr::kappa2 ---\n")
    print(irr::kappa2(data.frame(r1 = r1, r2 = r2), weight = "equal"))
    print(irr::kappa2(data.frame(r1 = r1, r2 = r2), weight = "squared"))
  }
}
