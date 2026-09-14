# DerSimonian-Laird Random-Effects (Reference Sec 47.269)
# Native R via metafor / meta; Python via PythonMeta / from-scratch.
# Run with:  Rscript dersimonian_laird_random_effects.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metafor::rma(method='DL')  -- classical DL RE meta-analysis\n")
  cat("  meta::metagen              -- generic RE meta-analysis with DL, REML, etc.\n")
  cat("  robumeta                   -- robust variance meta-regression\n")
  cat("Python:\n")
  cat("  PythonMeta / pymeta\n")
  cat("  From-scratch (see dersimonian_laird_random_effects.py)\n")
  cat("Refs: DerSimonian, R. & Laird, N. (1986) 'Meta-analysis in clinical\n")
  cat("      trials', Controlled Clin Trials 7(3), 177-188.\n")
}
