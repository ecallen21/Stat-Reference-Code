# Zig-Zag process sampler (Bierkens-Fearnhead-Roberts 2019)
# R: `RZigZag::ZigZag`, `PDMPFlux` (also supports zig-zag and BPS)
# Python: `pdmp_jax`, from-scratch
#
# library(RZigZag)
# fit <- ZigZagLogistic(X, y, n_epochs = 100)
# # Or the multivariate normal version:
# fit <- ZigZagGaussian(mu = rep(0, 5), covariance = diag(5),
#                       n_events = 5000)
