# Factorization Machines (Reference Sec 47.128)
# Native R via libFMR / FactoRizationMachines; Python via xLearn / pywFM / libFM.
# Run with:  Rscript factorization_machines.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  libFMR                       -- libFM R wrapper\n")
  cat("  FactoRizationMachines        -- pure-R implementation\n")
  cat("  h2o::h2o.deepwater / xgboost -- competing tabular baselines\n")
  cat("Python:\n")
  cat("  xLearn                       -- FM / FFM / LR at scale\n")
  cat("  pywFM                        -- libFM Python bindings (MCMC / ALS / SGD)\n")
  cat("  fastFM                       -- classic Python FM library\n")
  cat("  from-scratch                 -- see factorization_machines.py\n")
  cat("Refs: Rendle (2010) 'Factorization machines', ICDM.\n")
}
