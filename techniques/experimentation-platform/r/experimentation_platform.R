# Experimentation platform primitives (Reference Sec 44.11)
# Native: no dedicated R package; custom + stats::chisq.test.
# Run with:  Rscript experimentation_platform.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::chisq.test               -- SRM chi^2 test\n")
  cat("  digest                            -- deterministic hashing (md5, sha)\n")
  cat("  pwr                               -- power planning input to platform\n")
  cat("Python:\n")
  cat("  planout                           -- Facebook's experimentation DSL\n")
  cat("  scipy.stats.chi2                  -- SRM tests\n")
  cat("  eppo-sdk / statsig / growthbook   -- commercial experimentation SDKs\n")
  cat("Refs: Kohavi, Tang & Xu (2020) Trustworthy Online Controlled Experiments, CUP,\n")
  cat("      Ch 3-5, 22; Fabijan, Dmitriev, Olsson & Bosch (2017) 'The evolution of\n")
  cat("      continuous experimentation in software product development', ICSE.\n")
}
