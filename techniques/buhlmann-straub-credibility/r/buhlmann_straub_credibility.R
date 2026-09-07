# Buhlmann-Straub credibility (Reference Sec 24.21)
# Native R via actuar::cm(); Python via chainladder + from-scratch.
# Run with:  Rscript buhlmann_straub_credibility.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  actuar::cm(model = 'Buhlmann-Straub')  -- reference implementation\n")
  cat("  ChainLadder + credibility helpers\n")
  cat("  lme4::lmer with weights                -- BLUP equivalent\n")
  cat("Python:\n")
  cat("  chainladder (Cape Cod / BF / cred)\n")
  cat("  From-scratch numpy (see buhlmann_straub_credibility.py)\n")
  cat("Refs: Buhlmann, H. & Straub, E. (1970) 'Glaubwurdigkeit fur Schadensatze',\n")
  cat("      Bulletin ASA 70(1); Buhlmann & Gisler (2005) A Course in Credibility\n")
  cat("      Theory, Springer.\n")
}
