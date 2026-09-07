# Slice sampler (Reference Sec 25.11)
# Native R via mcmc, nimble; Python via pymc / from-scratch.
# Run with:  Rscript slice_sampler.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mcmc::slice.sample       -- 1-D slice + shrinkage\n")
  cat("  nimble builds slice steps into runMCMC\n")
  cat("  MCMCglmm / rstan (slice steps for constrained params)\n")
  cat("Python:\n")
  cat("  pymc.step_methods.Slice   -- discrete or continuous 1-D per param\n")
  cat("  numpyro slice sampler (community port)\n")
  cat("  from-scratch numpy (see slice_sampler.py)\n")
  cat("Refs: Neal, R.M. (2003) 'Slice sampling', Ann Stat 31(3): 705-767;\n")
  cat("      Damlen, P.J., Wakefield, J.C. & Walker, S.G. (1999) 'Gibbs\n")
  cat("      sampling for Bayesian non-conjugate and hierarchical models by\n")
  cat("      using auxiliary variables', JRSS-B.\n")
}
