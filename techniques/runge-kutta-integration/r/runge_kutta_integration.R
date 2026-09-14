# Runge-Kutta integration (Runge 1895; Kutta 1901)
# R: `deSolve::ode(method='rk4')` (Soetaert-Petzoldt-Setzer),
#    `pracma::rk4`, `deSolve` also has RKF45 (Cash-Karp)
# Python: `scipy.integrate.solve_ivp(method='RK45')` (default),
#         `scipy.integrate.odeint`, `torchdiffeq`, from-scratch
#
# library(deSolve)
# harm <- function(t, y, parms) list(c(y[2], -y[1]))
# sol <- ode(y = c(1, 0), times = seq(0, 10, 0.01),
#            func = harm, parms = NULL, method = "rk4")
