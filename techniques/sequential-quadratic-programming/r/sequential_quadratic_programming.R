# Sequential Quadratic Programming (Wilson 1963; Han 1976; Powell 1978)
# R: `nloptr::slsqp`, `Rsolnp::solnp` (augmented Lagrangian
#    with QP subproblems), `alabama::auglag`
# Python: `scipy.optimize.minimize(method='SLSQP')`,
#         `cvxpy` (QP subproblems), `pyipopt`, `casadi`,
#         from-scratch
#
# library(nloptr)
# slsqp(x0 = c(0, 1),
#       fn = function(x) 100*(x[2]-x[1]^2)^2 + (1-x[1])^2,
#       heq = function(x) x[1]^2 + x[2]^2 - 1)
