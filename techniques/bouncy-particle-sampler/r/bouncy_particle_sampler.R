# Bouncy Particle Sampler (Bouchard-Cote et al 2018 JASA)
# R: `RZigZag::BouncyParticle`, `PDMPFlux`
# Python: `pdmp_jax`, `bouncy` (custom), from-scratch
#
# library(RZigZag)
# fit <- BouncyParticle(dim = 3, mu = rep(0, 3),
#                       Sigma = matrix(c(2, 0.7, 0.3,
#                                        0.7, 1.5, 0.4,
#                                        0.3, 0.4, 1.0), 3),
#                       n_epochs = 100)
