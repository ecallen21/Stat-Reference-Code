# Egger Test for Publication Bias (Reference Sec 47.271)
# Native R via metafor / meta; Python via PythonMeta / from-scratch.
# Run with:  Rscript egger_test_publication_bias.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metafor::regtest         -- Egger's regression test\n")
  cat("  meta::metabias(method='linreg') -- Egger + variants (Peters, Begg)\n")
  cat("  metasens                 -- publication-bias sensitivity toolkit\n")
  cat("Python:\n")
  cat("  PythonMeta (partial)\n")
  cat("  From-scratch (see egger_test_publication_bias.py)\n")
  cat("Refs: Egger, M., Davey Smith, G., Schneider, M. & Minder, C. (1997)\n")
  cat("      'Bias in meta-analysis detected by a simple, graphical test',\n")
  cat("      BMJ 315(7109), 629-634.\n")
}
