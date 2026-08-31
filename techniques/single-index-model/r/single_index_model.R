# Single-index model (Reference Sec 33.8)
# Native R via np / SemiPar; Python via reticulate.
# Run with:  Rscript single_index_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  np                           -- Racine's nonparametric library (npindex, sim)\n")
  cat("  SemiPar                      -- Ruppert / Wand / Carroll semiparametric regression\n")
  cat("  gam, mgcv                    -- fixed-index generalised additive model (adjacent)\n")
  cat("Python:\n")
  cat("  sisreg                       -- single-index / multi-index regression\n")
  cat("  scikit-learn KernelRidge      -- kernel smoothing for the link g\n")
  cat("Refs: Ichimura, H. (1993) 'Semiparametric least squares (SLS) and weighted SLS\n")
  cat("      estimation of single-index models', J. Econometrics;\n")
  cat("      Hardle, W., Hall, P. & Ichimura, H. (1993) 'Optimal Smoothing in Single-Index\n")
  cat("      Models', Annals of Statistics.\n")
}
