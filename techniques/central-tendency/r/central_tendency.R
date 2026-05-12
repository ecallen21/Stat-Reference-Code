# Measures of central tendency (Reference §1.1)
#
# From-scratch implementations in base R, plus the idiomatic package calls.
# Run with:  Rscript central_tendency.R

# ---------------------------------------------------------------------------
# From scratch
# ---------------------------------------------------------------------------
arithmetic_mean <- function(x) sum(x) / length(x)

median_scratch <- function(x) {
  s <- sort(x); n <- length(s); mid <- (n + 1) / 2
  if (n %% 2 == 1) s[mid] else (s[n / 2] + s[n / 2 + 1]) / 2
}

mode_scratch <- function(x) {
  tab <- table(x)
  as.numeric(names(tab)[tab == max(tab)])   # may return >1 value
}

trimmed_mean_scratch <- function(x, proportion = 0.1) {
  stopifnot(proportion >= 0, proportion < 0.5)
  s <- sort(x); n <- length(s); k <- floor(n * proportion)
  kept <- if (k > 0) s[(k + 1):(n - k)] else s
  mean(kept)
}

winsorized_mean_scratch <- function(x, proportion = 0.1) {
  stopifnot(proportion >= 0, proportion < 0.5)
  s <- sort(x); n <- length(s); k <- floor(n * proportion)
  if (k > 0) {
    s[seq_len(k)] <- s[k + 1]
    s[(n - k + 1):n] <- s[n - k]
  }
  mean(s)
}

weighted_mean_scratch <- function(x, w) sum(x * w) / sum(w)

geometric_mean_scratch <- function(x) {
  stopifnot(all(x > 0))
  exp(mean(log(x)))
}

harmonic_mean_scratch <- function(x) {
  stopifnot(all(x > 0))
  length(x) / sum(1 / x)
}

# ---------------------------------------------------------------------------
# Library equivalents
#   base: mean(), median(), mean(x, trim=), weighted.mean()
#   psych: geometric.mean(), harmonic.mean(), winsor.mean()
# ---------------------------------------------------------------------------
library_versions <- function(x) {
  out <- list(
    `mean (base)`            = mean(x),
    `median (base)`          = median(x),
    `trimmed mean 20% (base)`= mean(x, trim = 0.2),
    `weighted.mean (base)`   = weighted.mean(x, w = seq_along(x))
  )
  if (requireNamespace("psych", quietly = TRUE)) {
    out$`geometric.mean (psych)` <- psych::geometric.mean(x)
    out$`harmonic.mean (psych)`  <- psych::harmonic.mean(x)
    out$`winsor.mean 20% (psych)`<- psych::winsor.mean(x, trim = 0.2)
  } else {
    out$`geometric.mean` <- geometric_mean_scratch(x)
    out$`harmonic.mean`  <- harmonic_mean_scratch(x)
  }
  out
}

# ---------------------------------------------------------------------------
if (sys.nframe() == 0) {
  data <- c(2, 4, 4, 4, 5, 5, 7, 9, 100)
  cat("data:", data, "\n\n--- from scratch ---\n")
  cat("arithmetic mean :", arithmetic_mean(data), "\n")
  cat("median          :", median_scratch(data), "\n")
  cat("mode            :", mode_scratch(data), "\n")
  cat("trimmed mean 20%:", trimmed_mean_scratch(data, 0.2), "\n")
  cat("winsor. mean 20%:", winsorized_mean_scratch(data, 0.2), "\n")
  cat("weighted mean   :", weighted_mean_scratch(data, seq_along(data)), "\n")
  cat("geometric mean  :", geometric_mean_scratch(data), "\n")
  cat("harmonic mean   :", harmonic_mean_scratch(data), "\n\n--- library ---\n")
  lv <- library_versions(data)
  for (nm in names(lv)) cat(sprintf("%-28s: %s\n", nm, lv[[nm]]))
}
