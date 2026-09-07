# Bridge sampling for marginal likelihood / evidence (Reference Sec 25.7)
# Native R via bridgesampling (Gronau); Python via arviz + custom.
# Run with:  Rscript bridge_sampling_evidence.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  bridgesampling::bridge_sampler  -- Gronau et al. one-liner from stan/JAGS/nimble output\n")
  cat("  BayesFactor                     -- ratios via Gaussian priors\n")
  cat("  loo::loo (WAIC / PSIS-LOO)      -- different criterion for model choice\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see bridge_sampling_evidence.py)\n")
  cat("  arviz for posterior samples (no direct bridge helper as of 2024)\n")
  cat("  numpyro / pymc + custom log-density evaluation on target + proposal\n")
  cat("Refs: Meng, X.-L. & Wong, W.H. (1996) 'Simulating ratios of normalising\n")
  cat("      constants via a simple identity: a theoretical exploration',\n")
  cat("      Statistica Sinica 6(4); Gronau, Q.F. et al. (2017) 'A tutorial on\n")
  cat("      bridge sampling', J Math Psychol 81: 80-97.\n")
}
