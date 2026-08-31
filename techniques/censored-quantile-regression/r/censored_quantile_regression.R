# Censored quantile regression (Reference Sec 33.3)
# Native R via quantreg::crq; Python via reticulate.
# Run with:  Rscript censored_quantile_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quantreg::crq               -- Powell (1986) + Portnoy (2003) censored QR\n")
  cat("  survival + quantreg         -- pair for tobit-style censoring + QR\n")
  cat("Python:\n")
  cat("  statsmodels QuantReg + custom censoring loop\n")
  cat("  pypowell / powell-crq        -- niche Python implementations\n")
  cat("  survival R via reticulate     -- best cross-check\n")
  cat("Refs: Powell, J.L. (1986) 'Censored regression quantiles', J. Econometrics;\n")
  cat("      Portnoy, S. (2003) 'Censored regression quantiles', JASA.\n")
}
