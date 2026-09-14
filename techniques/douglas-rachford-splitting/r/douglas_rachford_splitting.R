# Douglas-Rachford splitting (Douglas-Rachford 1956;
# Combettes-Wajs 2005)
# R: `CVXR` (uses DR-style splitting internally),
#    `flare` for TV / graph problems, custom loop.
# Python: `pyproximal.DouglasRachford`, `proxTV`,
#         `cvxpy` (ECOS/SCS use splitting), from-scratch
#
# library(CVXR)
# x <- Variable(100)
# obj <- Minimize(0.5 * sum_squares(A %*% x - b) + 0.5 * p_norm(x, 1))
# solve(Problem(obj), solver = "ECOS")
