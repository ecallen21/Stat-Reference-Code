# Differential Evolution (Storn-Price 1997)
# R: `DEoptim::DEoptim` (fast C++), `RcppDE`
# Python: `scipy.optimize.differential_evolution`, from-scratch
#
# library(DEoptim)
# rastr <- function(x) 10 * length(x) +
#   sum(x^2 - 10 * cos(2 * pi * x))
# fit <- DEoptim(rastr, lower = rep(-5.12, 5), upper = rep(5.12, 5),
#                control = DEoptim.control(NP = 30, itermax = 300))
