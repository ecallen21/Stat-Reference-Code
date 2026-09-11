# Stepped-Wedge Cluster Trial (Reference Sec 47.265)
# Native R via swCRTdesign / SWSamp / lmerTest; Python via statsmodels / from-scratch.
# Run with:  Rscript stepped_wedge_design.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  swCRTdesign::swPwr        -- power for SW-CRT under fixed period effects\n")
  cat("  SWSamp                    -- sample size / power for SW-CRT\n")
  cat("  lmerTest::lmer            -- fit mixed-effects analysis model\n")
  cat("  geepack::geeglm           -- GEE alternative with exchangeable working corr\n")
  cat("Python:\n")
  cat("  statsmodels.mixed_linear_model (MixedLM)\n")
  cat("  linearmodels.PanelOLS (cluster fixed effects)\n")
  cat("  From-scratch OLS (see stepped_wedge_design.py)\n")
  cat("Refs: Hussey, M.A. & Hughes, J.P. (2007) 'Design and analysis of\n")
  cat("      stepped wedge cluster randomized trials',\n")
  cat("      Contemp Clin Trials 28(2), 182-191.\n")
}
