# Genetic Algorithm (Reference Sec 47.141).
#
# Holland 1975 / Goldberg 1989. Selection + crossover + mutation
# on populations of candidate solutions.
#
# Uses `GA` (Scrucca) for binary / real-valued / permutation GAs.

if (!requireNamespace("GA", quietly = TRUE)) install.packages("GA")
library(GA)                                           # genetic algorithm engine

set.seed(0)
n <- 30
weights <- sample(1:20, n, replace = TRUE)
values <- sample(1:30, n, replace = TRUE)
capacity <- as.integer(0.4 * sum(weights))

fitness <- function(x) {
    w <- sum(weights * x); v <- sum(values * x)
    if (w > capacity) return(-Inf)
    v
}

ga <- ga(type = "binary", fitness = fitness, nBits = n,
          popSize = 60, maxiter = 500, run = 200, pcrossover = 0.8, pmutation = 0.03)
cat("=== GA on 0/1 knapsack (Holland 1975; Goldberg 1989) ===\n")
sol <- ga@solution[1, ]
cat(sprintf("  value = %d   weight = %d / %d   items picked = %d\n",
            sum(values * sol), sum(weights * sol), capacity, sum(sol)))
