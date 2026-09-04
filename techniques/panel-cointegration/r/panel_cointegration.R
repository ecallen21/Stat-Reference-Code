# Panel cointegration (Reference Sec 35.26)
# Native R via plm / punitroots; Python via arch / egcm.
# Run with:  Rscript panel_cointegration.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  plm::pcointtest              -- Pedroni + Westerlund panel cointegration\n")
  cat("  punitroots                    -- panel unit-root tests (Im-Pesaran-Shin, ...)\n")
  cat("  urca                          -- Johansen + adjacent single-series tests\n")
  cat("Python:\n")
  cat("  arch.unitroot                 -- ADF, PP, KPSS on residuals (single series)\n")
  cat("  egcm                           -- Engle-Granger cointegration (single series)\n")
  cat("  linearmodels.panel            -- panel infrastructure (no built-in Pedroni)\n")
  cat("Refs: Pedroni, P. (1999) 'Critical values for cointegration tests in\n")
  cat("      heterogeneous panels with multiple regressors', Oxford Bull Econ Stat;\n")
  cat("      Westerlund, J. (2007) 'Testing for error correction in panel data',\n")
  cat("      Oxford Bull Econ Stat; Pesaran, Shin & Smith (1999) 'Pooled mean group\n")
  cat("      estimation of dynamic heterogeneous panels', JASA.\n")
}
