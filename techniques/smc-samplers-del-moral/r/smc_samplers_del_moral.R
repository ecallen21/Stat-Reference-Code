# SMC samplers (Del Moral-Doucet-Jasra 2006)
# R: `SMC::sequential_MC`, `particles::compact_particle_filter`,
#    `nimbleSMC` (adds SMC step to NIMBLE MCMC).
# Python: `particles` (Chopin), `blackjax.tempered_smc`,
#         `pyfilter`, from-scratch
#
# library(SMC)
# fit <- sequential_MC(prior_sampler, log_lik, mh_kernel,
#                      lambda_schedule = seq(0, 1, length = 20),
#                      n_particles = 1000)
