# Interior-point / barrier method (Karmarkar 1984;
# Nesterov-Nemirovski 1994)
# R: `lpSolve::lp` (simplex, but interior-point via `highs`),
#    `Rglpk::Rglpk_solve_LP` (GLPK simplex/ipm),
#    `CVXR` (disciplined convex, interior-point backends),
#    `quadprog::solve.QP` (QP primal-dual)
# Python: `scipy.optimize.linprog(method='highs-ipm' | 'interior-point')`,
#         `cvxpy` (ECOS/SCS/Clarabel — all IPM), from-scratch
#
# library(CVXR)
# x <- Variable(2)
# obj <- Minimize(t(c(-3, -5)) %*% x)
# constr <- list(x[1] <= 4, x[2] <= 6, 3*x[1] + 2*x[2] <= 18, x >= 0)
# solve(Problem(obj, constr))
