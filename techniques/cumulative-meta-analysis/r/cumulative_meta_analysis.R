# Cumulative Meta-Analysis (Reference Sec 47.274)
# Native R via metafor / meta; Python via PythonMeta / from-scratch.
# Run with:  Rscript cumulative_meta_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metafor::cumul            -- cumulative meta-analysis over any ordering\n")
  cat("  meta::metacum             -- cumulative + Forest plot\n")
  cat("  dmetar::EvidenceCumulPlot -- publication-year evidence accumulation\n")
  cat("Python:\n")
  cat("  PythonMeta (partial)\n")
  cat("  From-scratch (see cumulative_meta_analysis.py)\n")
  cat("Refs: Lau, J., Antman, E.M., Jimenez-Silva, J., Kupelnick, B.,\n")
  cat("      Mosteller, F. & Chalmers, T.C. (1992) 'Cumulative meta-analysis\n")
  cat("      of therapeutic trials for myocardial infarction',\n")
  cat("      N Engl J Med 327(4), 248-254.\n")
}
