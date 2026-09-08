# Sparse Gaussian Process (Reference Sec 47.89)
# R has limited SGP; Python via GPflow / gpytorch.
# Run with:  Rscript sparse_gaussian_process.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mlegp                          -- exact GP + likelihood-based hyperpar MLE\n")
  cat("  kernlab::gausspr               -- exact GP regression / classification\n")
  cat("  DiceKriging                    -- geostat GPs; approx via nearest neighbours\n")
  cat("  reticulate + GPflow            -- call Python GPflow from R for SVGP\n")
  cat("Python:\n")
  cat("  GPflow (TensorFlow)            -- SVGP, FITC, deep GPs\n")
  cat("  gpytorch                       -- SVGP / DKL via PyTorch\n")
  cat("  scikit-learn.gaussian_process  -- exact GP baseline\n")
  cat("  from-scratch                   -- see sparse_gaussian_process.py\n")
  cat("Refs: Snelson & Ghahramani (2005) NeurIPS; Titsias (2009) AISTATS;\n")
  cat("      Hensman, Fusi & Lawrence (2013) 'Gaussian processes for big data', UAI.\n")
}
