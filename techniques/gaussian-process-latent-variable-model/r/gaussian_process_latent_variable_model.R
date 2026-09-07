# Gaussian process latent variable model (GPLVM) (Reference Sec 6.16)
# Deep-Bayesian dim-red; Python via GPy / gpflow / gpytorch.
# Run with:  Rscript gaussian_process_latent_variable_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  kergp                 -- Gaussian-process regression + custom kernels\n")
  cat("  DiceKriging           -- kriging / GP (regression baseline)\n")
  cat("  No dedicated GPLVM CRAN package as of 2024\n")
  cat("Python:\n")
  cat("  GPy.models.BCGPLVM / GPLVM / SparseGPLVM (Lawrence group)\n")
  cat("  gpflow.models.GPLVM  -- TensorFlow-backed GPLVM\n")
  cat("  gpytorch (Bayesian GPLVM via variational inference)\n")
  cat("  From-scratch numpy (see gaussian_process_latent_variable_model.py)\n")
  cat("Refs: Lawrence, N.D. (2004) 'Gaussian process latent variable models for\n")
  cat("      visualisation of high-dimensional data', NIPS; Titsias & Lawrence\n")
  cat("      (2010) 'Bayesian Gaussian process latent variable model', AISTATS.\n")
}
