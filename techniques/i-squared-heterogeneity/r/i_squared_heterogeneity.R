# I^2 Heterogeneity (Reference Sec 47.272)
# Native R via metafor / meta; Python via PythonMeta / from-scratch.
# Run with:  Rscript i_squared_heterogeneity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metafor::rma              -- reports Q, I^2, H^2, tau^2\n")
  cat("  meta::metagen             -- similar with prediction interval\n")
  cat("  dmetar::pcurve            -- p-curve as complement to heterogeneity\n")
  cat("Python:\n")
  cat("  PythonMeta / pymeta\n")
  cat("  From-scratch (see i_squared_heterogeneity.py)\n")
  cat("Refs: Higgins, J.P.T. & Thompson, S.G. (2002) 'Quantifying\n")
  cat("      heterogeneity in a meta-analysis', Stat Med 21(11), 1539-1558;\n")
  cat("      Higgins, J.P.T. et al (2003) 'Measuring inconsistency in\n")
  cat("      meta-analyses', BMJ 327(7414), 557-560.\n")
}
