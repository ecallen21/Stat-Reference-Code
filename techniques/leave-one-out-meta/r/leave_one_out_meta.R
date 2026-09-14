# Leave-One-Out Meta-Analysis (Reference Sec 47.273)
# Native R via metafor / meta; Python via PythonMeta / from-scratch.
# Run with:  Rscript leave_one_out_meta.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metafor::leave1out         -- L1O sensitivity output object\n")
  cat("  meta::metainf              -- influence + L1O plot / summary\n")
  cat("  dmetar::InfluenceAnalysis  -- combined L1O + Baujat + outliers\n")
  cat("Python:\n")
  cat("  PythonMeta (partial)\n")
  cat("  From-scratch (see leave_one_out_meta.py)\n")
  cat("Refs: Common sensitivity technique - originally credited to a Cochrane\n")
  cat("      Reviewers' Handbook convention; formal treatment in Higgins &\n")
  cat("      Thompson (2004) Stat Med 23(11).\n")
}
