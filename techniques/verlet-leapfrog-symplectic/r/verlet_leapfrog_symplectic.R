# Velocity Verlet / Leapfrog (Verlet 1967)
# R: `deSolve::ode(method='leapfrog')` — limited symplectic
#    support; usually custom loop or `pracma::rk4` etc.
# Python: `astropy.integrate.leapfrog`, `scipy.integrate.solve_ivp`
#         with symplectic methods (not built in — use custom or
#         `sympy.mechanics`), `blackjax` HMC uses leapfrog under
#         the hood.
#
# leapfrog <- function(q0, p0, gradV, h, n_steps, M_inv = 1) {
#   q <- q0; p <- p0
#   for (i in seq_len(n_steps)) {
#     p_half <- p - 0.5 * h * gradV(q)
#     q <- q + h * (M_inv * p_half)
#     p <- p_half - 0.5 * h * gradV(q)
#   }
#   list(q = q, p = p)
# }
