# Fay-Herriot small-area estimation (Reference Sec 27.6)
# Native R via sae / JoSAE; Python via samplics / from-scratch.
# Run with:  Rscript fay_herriot_small_area.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  sae::mseFH / eblupFH        -- Fay-Herriot EBLUP + Prasad-Rao MSE\n")
  cat("  JoSAE                       -- unit and area-level SAE\n")
  cat("  emdi                        -- Bayesian small-area with poverty measures\n")
  cat("  BayesSAE                    -- Bayesian Fay-Herriot\n")
  cat("Python:\n")
  cat("  samplics.sae.eblupFayHerriot -- Python port of sae::eblupFH\n")
  cat("  pymc (custom hierarchical model)\n")
  cat("Refs: Fay, R.E. & Herriot, R.A. (1979) 'Estimates of income for small\n")
  cat("      places: an application of James-Stein procedures to census data',\n")
  cat("      JASA 74(366): 269-277; Rao & Molina (2015) Small Area Estimation,\n")
  cat("      2nd ed., Wiley.\n")
}
