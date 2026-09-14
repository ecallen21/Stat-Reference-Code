# Parallel Tempering / Replica Exchange (Geyer 1991)
# R: `nimble` (tempered MCMC in NIMBLE), `mcmcse`,
#    `parallel::mclapply` for chain-level parallelism.
# Python: `emcee.PTSampler` (deprecated but reference),
#         `pymc.step_methods.MLDA` (multi-level), `blackjax`
#         `tempered_smc` variants, from-scratch
#
# library(nimble)
# # NIMBLE supports parallel tempering via configureMCMC options
