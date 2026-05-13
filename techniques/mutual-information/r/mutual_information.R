# Mutual information as an association measure (Reference §4.14)
# From-scratch base R plus infotheo::mutinformation.
# Run with:  Rscript mutual_information.R
#
# Inputs used below:
#   x, y    : discrete (categorical / integer) vectors of equal length, OR
#             continuous vectors to be binned by bin_counts / cut()
#   base    : log base; 2 -> bits, exp(1) -> nats, 10 -> Hartleys

entropy_discrete_scratch <- function(x, base = 2) {
  n <- length(x); p <- table(x) / n
  -sum(p * log(p, base = base))
}

mutual_information_discrete_scratch <- function(x, y, base = 2) {
  stopifnot(length(x) == length(y))
  n <- length(x)
  joint <- table(x, y) / n
  px <- rowSums(joint); py <- colSums(joint)
  I <- 0
  for (i in seq_len(nrow(joint))) for (j in seq_len(ncol(joint))) {
    pxy <- joint[i, j]; if (pxy == 0) next
    I <- I + pxy * log(pxy / (px[i] * py[j]), base = base)
  }
  Hx <- entropy_discrete_scratch(x, base); Hy <- entropy_discrete_scratch(y, base)
  list(I_xy = unname(I), H_x = Hx, H_y = Hy,
       nmi_arith = unname(I / ((Hx + Hy) / 2)),
       nmi_geom  = unname(I / sqrt(Hx * Hy)),
       nmi_max   = unname(I / max(Hx, Hy)),
       base = base)
}

mutual_information_binned_scratch <- function(x, y, bins = NULL, base = 2) {
  if (is.null(bins)) {
    fd <- function(a) {
      h <- 2 * IQR(a) * length(a)^(-1/3); if (h == 0) h <- diff(range(a)) / 10
      max(2, ceiling(diff(range(a)) / h))
    }
    bx <- fd(x); by <- fd(y)
  } else { bx <- by <- bins }
  xb <- cut(x, breaks = bx, labels = FALSE, include.lowest = TRUE)
  yb <- cut(y, breaks = by, labels = FALSE, include.lowest = TRUE)
  out <- mutual_information_discrete_scratch(xb, yb, base = base)
  out$bins_x <- bx; out$bins_y <- by; out
}

# Library: infotheo::mutinformation, entropy::mi.empirical, mpmi::mmi
library_demo <- function(x, y) {
  if (requireNamespace("infotheo", quietly = TRUE)) {
    cat("infotheo::mutinformation:", infotheo::mutinformation(x, y), " nats\n")
    cat("infotheo::natstobits:    ",
        infotheo::natstobits(infotheo::mutinformation(x, y)), " bits\n")
  } else cat("(install 'infotheo' for mutinformation)\n")
}

if (sys.nframe() == 0) {
  set.seed(13)
  cat("=== Discrete: two correlated 3-level variables ===\n")
  x <- sample(0:2, 1000, replace = TRUE)
  y <- ifelse(runif(1000) < 0.7, x, sample(0:2, 1000, replace = TRUE))
  print(mutual_information_discrete_scratch(x, y))
  cat("\n=== Continuous (binned): y = x^2 + noise ===\n")
  xc <- rnorm(500); yc <- xc^2 + rnorm(500, 0, 0.2)
  print(mutual_information_binned_scratch(xc, yc))
  cat("Pearson r:", cor(xc, yc), "\n")
  cat("\n--- library ---\n"); library_demo(x, y)
}
