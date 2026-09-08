# Simulated Annealing (Reference Sec 47.140).
#
# Kirkpatrick, Gelatt & Vecchi 1983. Metropolis-based global
# optimizer with cooling schedule.
#
# Uses `GenSA` for generalised SA (Xiang et al) or base `optim(method="SANN")`.

if (!requireNamespace("GenSA", quietly = TRUE)) install.packages("GenSA")
library(GenSA)                                        # generalised simulated annealing

f1 <- function(x) x[1]^2 + 20 * sin(x[1])^2

set.seed(0)
res <- GenSA(par = -3.0, fn = f1, lower = -10, upper = 10,
              control = list(maxit = 5000, temperature = 5))
cat("=== GenSA on 1-D multi-well (Kirkpatrick-Gelatt-Vecchi 1983) ===\n")
cat(sprintf("  x_opt = %.6f (truth: x = 0)\n", res$par))
cat(sprintf("  f_opt = %.6f (truth: f = 0)\n", res$value))
