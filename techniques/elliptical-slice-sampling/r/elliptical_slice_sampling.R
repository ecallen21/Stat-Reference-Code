# Elliptical slice sampling (Reference Sec 25.12)
# Native R via ess / spBayes; Python via numpy / gpytorch.
# Run with:  Rscript elliptical_slice_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ess                          -- reference elliptical slice implementation\n")
  cat("  spBayes                       -- GP hierarchical Bayes uses ESS steps\n")
  cat("  greta / cmdstanr             -- alternative GP posteriors via HMC\n")
  cat("Python:\n")
  cat("  gpytorch (ESS as step method for likelihood-free GP variants)\n")
  cat("  From-scratch numpy (see elliptical_slice_sampling.py)\n")
  cat("  pymc: numeric implementations wrap Murray-Adams ESS\n")
  cat("Refs: Murray, I., Adams, R.P. & MacKay, D.J.C. (2010) 'Elliptical slice\n")
  cat("      sampling', AISTATS 9; Nishihara, R. et al. (2014) 'Parallel MCMC\n")
  cat("      with generalized elliptical slice sampling', JMLR.\n")
}
