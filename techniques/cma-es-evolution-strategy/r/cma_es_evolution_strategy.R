# CMA-ES (Reference Sec 47.138).
#
# Hansen & Ostermeier 2001. Population-based evolution strategy that
# adapts a full-covariance Gaussian search distribution.
#
# Uses `cmaes` (Trautmann/Mersmann) or `cmaesr`.

if (!requireNamespace("cmaes", quietly = TRUE)) install.packages("cmaes")
library(cmaes)                                        # CMA-ES engine

rosen <- function(x) sum(100 * (x[-1] - x[-length(x)]^2)^2 + (1 - x[-length(x)])^2)

set.seed(0)
x0 <- runif(5, -2, 2)
res <- cma_es(par = x0, fn = rosen,
              control = list(mu = 4, lambda = 12, maxit = 500))
cat("=== CMA-ES on 5-D Rosenbrock (Hansen-Ostermeier 2001) ===\n")
cat(sprintf("  x_opt = %s\n", toString(round(res$par, 4))))
cat(sprintf("  f_opt = %.6f (truth: x = 1, f = 0)\n", res$value))
