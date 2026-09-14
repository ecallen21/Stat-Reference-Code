# Huber M-estimator (Huber 1964)
# R: `MASS::rlm(method='M')` (Huber default),
#    `robustbase::lmrob`, `robust::lmRob`
# Python: `sklearn.linear_model.HuberRegressor`,
#         `statsmodels.robust.robust_linear_model.RLM`,
#         from-scratch IRLS
#
# library(MASS)
# fit <- rlm(y ~ x1 + x2 + x3, data = df, method = "M",
#            psi = psi.huber, k = 1.345)
