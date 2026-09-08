# Particle Swarm Optimization (Reference Sec 47.139).
#
# Kennedy & Eberhart 1995. Population-based global optimizer using
# swarm dynamics (velocity toward personal & global best).
#
# Uses `pso` (Bendtsen) or `psoptim`.

if (!requireNamespace("pso", quietly = TRUE)) install.packages("pso")
library(pso)                                          # PSO engine

rastrigin <- function(x) 10 * length(x) + sum(x^2 - 10 * cos(2 * pi * x))

set.seed(0)
res <- psoptim(par = rep(0, 5),
                fn = rastrigin,
                lower = rep(-5.12, 5), upper = rep(5.12, 5),
                control = list(maxit = 300, s = 30))
cat("=== PSO on 5-D Rastrigin (Kennedy-Eberhart 1995) ===\n")
cat(sprintf("  x_opt = %s\n", toString(round(res$par, 4))))
cat(sprintf("  f_opt = %.6f (truth: x = 0, f = 0)\n", res$value))
