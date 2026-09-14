# Basin hopping (Wales-Doye 1997)
# R: `GenSA::GenSA` (generalised simulated annealing, similar
#    hybrid), `smoof`, custom loop with `optim` + perturbation
# Python: `scipy.optimize.basinhopping`, from-scratch
#
# library(GenSA)
# fit <- GenSA(par = c(3.5, -2.0), fn = rastr,
#              lower = rep(-5.12, 2), upper = rep(5.12, 2),
#              control = list(max.time = 5))
