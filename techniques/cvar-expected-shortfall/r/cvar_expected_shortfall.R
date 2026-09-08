# CVaR / Expected Shortfall (Reference Sec 47.42)
# Native R via PerformanceAnalytics; Python via empyrical / cvxpy.
# Run with:  Rscript cvar_expected_shortfall.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  PerformanceAnalytics::ES / VaR    -- historical / Gaussian / modified\n")
  cat("  quantreg::rq  + Koenker-Bassett   -- CVaR-regression via RU\n")
  cat("  qrmtools                          -- QRM textbook helpers\n")
  cat("Python:\n")
  cat("  empyrical.cvar / value_at_risk    -- financial-metrics library\n")
  cat("  cvxpy   (RU LP form)              -- convex portfolio optimisation\n")
  cat("  from-scratch                      -- see cvar_expected_shortfall.py\n")
  cat("Refs: Rockafellar & Uryasev (2000) J Risk 2(3); Artzner, Delbaen, Eber,\n")
  cat("      Heath (1999) 'Coherent measures of risk', Math Finance 9(3).\n")
}
