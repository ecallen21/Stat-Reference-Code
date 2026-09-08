# Thompson sampling (Reference Sec 47.60)
# Native R via contextual; Python via scikit-multiflow / mabwiser.
# Run with:  Rscript thompson_sampling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  contextual::ThompsonSamplingPolicy -- Bernoulli / Gaussian / contextual TS\n")
  cat("  bandit                             -- simple MAB simulators\n")
  cat("  bayesbandits                       -- Bayesian bandits toolkit\n")
  cat("Python:\n")
  cat("  mabwiser.MAB(policy='ThompsonSampling')  -- MAB library with TS\n")
  cat("  scikit-multiflow.meta.OnlineBoosting     -- streaming bandit variants\n")
  cat("  vowpalwabbit --contextual_bandit         -- contextual TS at scale\n")
  cat("  from-scratch                             -- see thompson_sampling.py\n")
  cat("Refs: Thompson (1933) Biometrika 25; Agrawal & Goyal (2012) COLT;\n")
  cat("      Russo, Van Roy, Kazerouni, Osband & Wen (2018) 'A tutorial on\n")
  cat("      Thompson sampling', FnT ML 11(1).\n")
}
