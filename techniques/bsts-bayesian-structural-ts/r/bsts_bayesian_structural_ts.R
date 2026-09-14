# BSTS — Bayesian Structural Time Series (Scott-Varian 2014)
# R: `bsts::bsts` (Steven Scott), `KFAS::KFS` (state space MLE),
#    `dlm::dlmMLE`, `tsibble` + `fable::ETS`
# Python: `pybsts`, `sts-anomaly`, `statsmodels.tsa.UnobservedComponents`,
#         `pyro` state-space, from-scratch
#
# library(bsts)
# ss <- AddLocalLinearTrend(list(), y = y)
# ss <- AddSeasonal(ss, y = y, nseasons = 12)
# fit <- bsts(y, state.specification = ss, niter = 1000)
# pred <- predict(fit, horizon = 12)
