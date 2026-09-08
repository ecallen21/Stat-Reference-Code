# Evolution Strategies (OpenAI, Reference Sec 47.147).
#
# Salimans et al 2017. Antithetic finite-difference gradient of
# smoothed objective. Rank-shaped update for scale invariance.
#
# No R package for OpenAI-flavour ES; a minimal from-scratch impl.

rank_scores <- function(f) {
    n <- length(f)
    r <- rank(f, ties.method = "average") - 1
    r / (n - 1) - 0.5
}

openai_es <- function(f, x0, sigma = 0.1, lr = 0.03, pop = 40, n_iter = 200) {
    x <- x0
    d <- length(x)
    for (it in seq_len(n_iter)) {
        eps <- matrix(rnorm((pop / 2) * d), pop / 2, d)
        eps <- rbind(eps, -eps)
        fits <- apply(eps, 1, function(e) f(x + sigma * e))
        r <- rank_scores(-fits)
        grad <- crossprod(eps, r) / (pop * sigma)
        x <- x + lr * as.numeric(grad)
    }
    list(x = x, f = f(x))
}

set.seed(0)
bowl <- function(x) sum(x^2)
r <- openai_es(bowl, rep(3.0, 10), sigma = 0.5, lr = 0.5, pop = 40, n_iter = 300)
cat("=== OpenAI ES on 10-D bowl (Salimans et al 2017) ===\n")
cat(sprintf("  f_final = %.6f   ||x||_inf = %.4f\n",
            r$f, max(abs(r$x))))
