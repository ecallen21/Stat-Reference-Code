# MANOVA (Reference §9.2)
# From-scratch base R plus stats::manova (idiomatic wrapper).
# Run with:  Rscript manova.R
#
# Inputs:
#   groups : list of n_k x p numeric matrices, one per group.

sscp_matrices <- function(groups) {
  X <- do.call(rbind, groups); N <- nrow(X); p <- ncol(X); K <- length(groups)
  grand <- colMeans(X)
  H <- matrix(0, p, p); E <- matrix(0, p, p)
  for (g in groups) {
    n_k <- nrow(g); m_k <- colMeans(g); d <- m_k - grand
    H <- H + n_k * (d %o% d)
    cent <- sweep(g, 2, m_k); E <- E + t(cent) %*% cent
  }
  list(H = H, E = E, N = N, p = p, K = K)
}

manova_scratch <- function(groups) {
  z <- sscp_matrices(groups); H <- z$H; E <- z$E
  N <- z$N; p <- z$p; K <- z$K
  M <- solve(E, H)
  lam <- sort(Re(eigen(M, only.values = TRUE)$values), decreasing = TRUE)
  lam <- pmax(lam, 0)
  nu_h <- K - 1; nu_e <- N - K; s <- min(p, nu_h)

  # Wilks
  wilks <- prod(1 / (1 + lam))
  t2 <- (p^2 * nu_h^2 - 4) / (p^2 + nu_h^2 - 5); tt <- sqrt(max(t2, 1))
  ms <- nu_e + nu_h - (p + nu_h + 1) / 2
  y <- wilks^(1 / tt)
  F_w <- ((1 - y) / y) * (ms * tt - p * nu_h / 2 + 1) / (p * nu_h)
  df1_w <- p * nu_h; df2_w <- ms * tt - p * nu_h / 2 + 1

  # Pillai
  pillai <- sum(lam / (1 + lam))
  m_p <- (abs(p - nu_h) - 1) / 2; n_p <- (nu_e - p - 1) / 2
  df1_p <- s * (2 * m_p + s + 1); df2_p <- s * (2 * n_p + s + 1)
  F_p <- ((2 * n_p + s + 1) / (2 * m_p + s + 1)) * (pillai / (s - pillai))

  # Hotelling-Lawley
  hl <- sum(lam)
  df1_hl <- s * (2 * m_p + s + 1); df2_hl <- 2 * (s * n_p + 1)
  F_hl <- (df2_hl / df1_hl) * hl / s

  # Roy
  roy <- max(lam); r <- max(p, nu_h); df2_r <- nu_e - r + nu_h
  F_roy <- (df2_r / r) * roy

  list(
    eigenvalues = lam,
    Wilks = list(stat = wilks, F = F_w, df1 = df1_w, df2 = df2_w,
                 p_value = pf(F_w, df1_w, df2_w, lower.tail = FALSE)),
    Pillai = list(stat = pillai, F = F_p, df1 = df1_p, df2 = df2_p,
                  p_value = pf(F_p, df1_p, df2_p, lower.tail = FALSE)),
    Hotelling_Lawley = list(stat = hl, F = F_hl, df1 = df1_hl, df2 = df2_hl,
                             p_value = pf(F_hl, df1_hl, df2_hl, lower.tail = FALSE)),
    Roy = list(stat = roy, F = F_roy, df1 = r, df2 = df2_r,
               p_value = pf(F_roy, r, df2_r, lower.tail = FALSE))
  )
}

if (sys.nframe() == 0) {
  set.seed(17)
  Sigma <- matrix(c(1, 0.3, 0.2, 0.3, 1, 0.4, 0.2, 0.4, 1), 3, 3)
  g1 <- MASS::mvrnorm(50, c( 0.0, 0.0,  0.0), Sigma)
  g2 <- MASS::mvrnorm(55, c( 0.5, 0.2, -0.3), Sigma)
  g3 <- MASS::mvrnorm(60, c(-0.3, 0.8,  0.4), Sigma)
  cat("=== From scratch ===\n"); print(manova_scratch(list(g1, g2, g3)))

  cat("\n--- library: stats::manova ---\n")
  Y <- rbind(g1, g2, g3)
  grp <- factor(c(rep("a", 50), rep("b", 55), rep("c", 60)))
  m <- manova(Y ~ grp)
  print(summary(m, test = "Wilks"))
  print(summary(m, test = "Pillai"))
  print(summary(m, test = "Hotelling-Lawley"))
  print(summary(m, test = "Roy"))
}
