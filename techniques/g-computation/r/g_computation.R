# G-computation / parametric G-formula (Reference Sec 15.20)
# Native R via stdReg / gfoRmula; Python zepid + custom.
# Run with:  Rscript g_computation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stdReg::stdGlm                    -- standardisation-based ATE from GLM\n")
  cat("  gfoRmula                            -- sequential g-formula for time-varying\n")
  cat("  ipw + custom                        -- alternative implementation\n")
  cat("Python:\n")
  cat("  zepid.causal.gformula.GFormula     -- parametric g-formula\n")
  cat("  causalinference                     -- Rubin-style causal-inference\n")
  cat("  sklearn + custom                   -- baseline\n")
  cat("Refs: Robins (1986) 'A new approach to causal inference in mortality studies',\n")
  cat("      Math Modelling; Hernán & Robins (2020) Causal Inference: What If,\n")
  cat("      Chapman & Hall/CRC, Ch 13.\n")
}
