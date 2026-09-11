# Best-of-N Sampling (Reference Sec 47.196).
#
# Cobbe 2021; Nakano 2021 WebGPT. Simple test-time scaling: sample
# N candidates, pick the one with highest reward-model score.

set.seed(0)

# Empirical E[max of N i.i.d. N(0,1)]
cat("=== Best-of-N Sampling (Cobbe 2021; Nakano 2021) ===\n")
cat("  E[max of N N(0,1)] (MC 20k reps + sqrt(2 ln N) asymptotic):\n")
for (N in c(1, 2, 4, 8, 16, 32, 64, 128)) {
    xs <- matrix(rnorm(20000 * N), 20000, N)
    mc <- mean(apply(xs, 1, max))
    theory <- sqrt(2 * log(max(N, 2)))
    cat(sprintf("    N = %4d   MC = %8.4f   sqrt(2 ln N) = %8.4f\n", N, mc, theory))
}
cat("\nFor real LLMs: transformers.generate(num_return_sequences=N) + custom reward model.\n")
