# Savage-Dickey Bayes factor (Reference Sec 47.59)
# Native R via BayesFactor / bridgesampling / polspline; Python via PyMC / arviz.
# Run with:  Rscript savage_dickey_bf.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  BayesFactor::ttestBF / lmBF   -- default priors + BF for common tests\n")
  cat("  bridgesampling::bridge_sampler-- bridge sampling + Savage-Dickey helpers\n")
  cat("  polspline::psknots / logspline-- density estimation at theta0\n")
  cat("Python:\n")
  cat("  pymc.stats.compute_log_likelihood + arviz.compare (LOO / WAIC)\n")
  cat("  pystan / cmdstanpy + custom prior/posterior density at theta0\n")
  cat("  from-scratch                  -- see savage_dickey_bf.py\n")
  cat("Refs: Dickey (1971) Ann Math Stat 42; Wagenmakers, Lodewyckx, Kuriyal &\n")
  cat("      Grasman (2010) Cognitive Psychology 60(3).\n")
}
