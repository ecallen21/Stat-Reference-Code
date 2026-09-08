# RIF (unconditional quantile) regression (Reference Sec 47.46)
# Native R via dineq / uqr; Python via statsmodels + custom.
# Run with:  Rscript rif_regression_firpo.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  dineq::rifr / rif.decompose  -- Firpo-Fortin-Lemieux RIF + decomposition\n")
  cat("  uqr::urq                     -- unconditional QR alternate implementation\n")
  cat("  quantreg::rq                 -- conditional QR baseline (Koenker-Bassett)\n")
  cat("Python:\n")
  cat("  statsmodels.QuantReg         -- conditional QR only; wrap for RIF\n")
  cat("  from-scratch                 -- see rif_regression_firpo.py\n")
  cat("Refs: Firpo, Fortin & Lemieux (2009) Econometrica 77(3); Firpo, Fortin &\n")
  cat("      Lemieux (2018) 'Decomposing wage distributions using RIF regressions',\n")
  cat("      Econometrics 6(2).\n")
}
