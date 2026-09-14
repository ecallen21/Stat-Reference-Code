# Trust-region optimisation (Powell 1970; Steihaug 1983; Byrd-Schnabel 2000)
# R: `trust::trust`, `optim(method='BFGS')` uses line-search
#    but `nloptr::nloptr(algorithm='NLOPT_LD_TRUSTREGION_...')`,
#    `pracma::trust.region` (partial)
# Python: `scipy.optimize.minimize(method='trust-ncg' | 'trust-krylov' | 'dogleg')`,
#         `scipy.optimize.least_squares(method='trf')`,
#         from-scratch
#
# library(trust)
# fit <- trust(objfun, parinit, rinit = 1, rmax = 10, iterlim = 100)
