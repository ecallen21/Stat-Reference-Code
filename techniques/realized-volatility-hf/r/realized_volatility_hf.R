# Realized volatility from HF data (Reference Sec 47.75)
# Native R via highfrequency; Python via arch / custom.
# Run with:  Rscript realized_volatility_hf.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  highfrequency::rCov / rBPCov / rTSCov       -- RV, BV, two-scales, MRK\n")
  cat("  highfrequency::rHARModel                    -- HAR-RV forecasting\n")
  cat("  RTAQ / xts                                  -- TAQ tick-data loaders\n")
  cat("Python:\n")
  cat("  arch                                        -- HARX + realized measures\n")
  cat("  MFE                                         -- Sheppard's MATLAB port\n")
  cat("  from-scratch                                -- see realized_volatility_hf.py\n")
  cat("Refs: Andersen & Bollerslev (1998) IER 39(4); Barndorff-Nielsen & Shephard\n")
  cat("      (2004) J Fin Econom 2(1); Zhang, Mykland & Ait-Sahalia (2005) JASA\n")
  cat("      100(472); Corsi (2009) 'HAR' J Fin Econom 7(2).\n")
}
