# Net Reclassification Improvement (NRI) + Integrated Discrimination Improvement (IDI)
# Reference §20.x extra.  Run with:  Rscript nri_idi.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Hmisc::improveProb(x1, x2, y)                -- NRI + IDI with bootstrap SEs\n")
  cat("  nricens::nribin(mdl.std, mdl.new, ...)       -- categorical / continuous NRI for binary\n")
  cat("  nricens::nricens(mdl.std, mdl.new, tt=...)    -- NRI for censored (survival) outcomes\n")
  cat("  survIDINRI::IDI.INF(pt1, pt2, y, indicator, t0)  -- IDI for survival outcomes\n")
  cat("  Warning: NRI has known statistical issues — see Pepe et al. 2014, Kerr et al. 2014.\n")
  cat("  Alternative: decision-curve-analysis (see repo).\n")
}
