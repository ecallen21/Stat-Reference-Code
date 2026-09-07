# Marginal Structural Model (Reference Sec 15.19)
# Native R via ipw + geeglm; Python zepid + custom.
# Run with:  Rscript marginal_structural_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ipw::ipwtm                        -- IPTW for time-varying treatments\n")
  cat("  geepack::geeglm                   -- GEE outcome model on weighted pseudo-pop\n")
  cat("  survival::coxph + weights          -- survival MSM\n")
  cat("Python:\n")
  cat("  zepid.causal.gformula.MSMIPTW    -- MSM with stabilised IPTW\n")
  cat("  causalinference                    -- Rubin-style tools\n")
  cat("  sklearn + custom\n")
  cat("Refs: Robins, Hernán & Brumback (2000) 'Marginal structural models and causal\n")
  cat("      inference in epidemiology', Epidemiology; Hernán & Robins (2020) Causal\n")
  cat("      Inference: What If, CRC, Ch 21.\n")
}
