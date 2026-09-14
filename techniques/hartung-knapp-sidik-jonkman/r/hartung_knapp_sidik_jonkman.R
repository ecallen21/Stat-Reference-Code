# Hartung-Knapp-Sidik-Jonkman (Reference Sec 47.270)
# Native R via metafor / meta; Python via PythonMeta / from-scratch.
# Run with:  Rscript hartung_knapp_sidik_jonkman.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metafor::rma(test='knha')  -- Knapp-Hartung adjustment for RE meta\n")
  cat("  meta::metagen(hakn=TRUE)   -- HKSJ variance and t-quantile CI\n")
  cat("  robumeta                   -- robust variance analogue\n")
  cat("Python:\n")
  cat("  PythonMeta (partial support)\n")
  cat("  From-scratch (see hartung_knapp_sidik_jonkman.py)\n")
  cat("Refs: Hartung, J. & Knapp, G. (2001) 'On tests of the overall treatment\n")
  cat("      effect in meta-analysis with normally distributed responses',\n")
  cat("      Stat Med 20(12), 1771-1782;  Sidik, K. & Jonkman, J.N. (2002)\n")
  cat("      'A simple confidence interval for meta-analysis',\n")
  cat("      Stat Med 21(21), 3153-3159.\n")
}
