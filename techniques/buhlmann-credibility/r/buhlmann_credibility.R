# Buhlmann credibility (Reference Sec 24.20)
# Native R via actuar / ChainLadder; Python via chainladder + from-scratch.
# Run with:  Rscript buhlmann_credibility.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  actuar::cm             -- credibility estimators (Buhlmann, B-Straub, hierarchical)\n")
  cat("  ChainLadder::MackChainLadder + credibility helpers\n")
  cat("  lme4::lmer (linear mixed model) -- BLUP equivalence to Buhlmann\n")
  cat("Python:\n")
  cat("  chainladder (from Actuarial Community) -- Cape Cod, Bornhuetter-Ferguson, cred\n")
  cat("  From-scratch numpy (see buhlmann_credibility.py)\n")
  cat("  statsmodels.MixedLM (as BLUP alternative)\n")
  cat("Refs: Buhlmann, H. (1967) 'Experience rating and credibility', ASTIN\n")
  cat("      Bulletin 4; Buhlmann & Gisler (2005) A Course in Credibility Theory\n")
  cat("      and its Applications, Springer.\n")
}
