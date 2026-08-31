# Jackknife+ prediction intervals (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python; native R alternatives listed.
# Run with:  Rscript jackknife_plus.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  conformalInference          -- jackknife+, CV+, split-conformal (Lei-Wasserman refit)\n")
  cat("  cfcausal                    -- weighted conformal for causal effects\n")
  cat("Python:\n")
  cat("  mapie.regression.MapieRegressor(method='plus') -- jackknife+ / CV+\n")
  cat("  nonconformist                -- classic inductive + jackknife conformal\n")
  cat("  puncc                       -- deel-ai conformal toolbox\n")
  cat("Refs: Barber, R.F., Candes, E.J., Ramdas, A. & Tibshirani, R.J. (2021)\n")
  cat("      'Predictive inference with the jackknife+', Annals of Statistics 49(1).\n")
}
