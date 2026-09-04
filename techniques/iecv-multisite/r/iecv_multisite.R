# Internal-external CV (Reference Sec 39.25)
# Native R via metamisc + rms; Python sklearn + custom.
# Run with:  Rscript iecv_multisite.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metamisc::valmeta              -- meta-analysis of per-site validation stats\n")
  cat("  rms::validate                  -- classic internal validation\n")
  cat("  pmsampsize                     -- sample-size planning for CPMs\n")
  cat("Python:\n")
  cat("  sklearn.model_selection.LeaveOneGroupOut\n")
  cat("  custom                         -- IECV loop + weighted-mean pooling\n")
  cat("Refs: Debray, T.P.A. et al. (2013) 'A framework for developing, implementing,\n")
  cat("      and evaluating clinical prediction models in an individual participant\n")
  cat("      data meta-analysis', Stat Med; Steyerberg, E.W. & Harrell, F.E. (2016) JCE.\n")
}
