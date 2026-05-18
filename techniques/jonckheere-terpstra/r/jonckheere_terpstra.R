# Jonckheere-Terpstra Test (Reference §6.10)
# From-scratch base R plus clinfun::jonckheere.test or DescTools::JonckheereTerpstraTest.
# Run with:  Rscript jonckheere_terpstra.R
#
# Inputs:  groups (list in assumed order under H_A), alternative

jonckheere_terpstra_scratch <- function(groups, alternative = "two.sided") {
  k <- length(groups); sizes <- vapply(groups, length, integer(1)); N <- sum(sizes)
  J <- 0
  for (i in seq_len(k - 1)) for (j in (i + 1):k) {
    gi <- groups[[i]]; gj <- groups[[j]]
    U_ij <- sum(outer(gi, gj, "<")) + 0.5 * sum(outer(gi, gj, "=="))
    J <- J + U_ij
  }
  mu <- (N^2 - sum(sizes^2)) / 4
  var <- (N^2 * (2 * N + 3) - sum(sizes^2 * (2 * sizes + 3))) / 72
  z <- (J - mu) / sqrt(var)
  p <- switch(alternative,
    "two.sided"  = 2 * pnorm(-abs(z)),
    "increasing" = pnorm(z, lower.tail = FALSE),
    "decreasing" = pnorm(z))
  list(J = J, expected_J = mu, var_J = var, z = z,
       p_value = p, n_per_group = sizes, alternative = alternative,
       method = "Jonckheere-Terpstra")
}

library_demo <- function(groups) {
  if (requireNamespace("clinfun", quietly = TRUE)) {
    values <- unlist(groups)
    grp <- factor(rep(seq_along(groups), sapply(groups, length)),
                  ordered = TRUE)
    print(clinfun::jonckheere.test(values, grp, alternative = "increasing"))
  } else cat("(install 'clinfun' for jonckheere.test)\n")
}

if (sys.nframe() == 0) {
  set.seed(7)
  g1 <- rnorm(25, 50, 5); g2 <- rnorm(25, 53, 5)
  g3 <- rnorm(25, 57, 5); g4 <- rnorm(25, 62, 5)
  print(jonckheere_terpstra_scratch(list(g1, g2, g3, g4), alternative = "increasing"))
  cat("\n--- library ---\n"); library_demo(list(g1, g2, g3, g4))
}
