# Factor mixture model (Reference Sec 36.10)
# Native R via OpenMx / MplusAutomation; Python custom.
# Run with:  Rscript factor_mixture_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  OpenMx                            -- confirmatory FMM specification\n")
  cat("  MplusAutomation                   -- FMM via Mplus (canonical)\n")
  cat("  lavaan (limited FMM)              -- SEM baseline\n")
  cat("  mixtools                           -- mixture EM (no factor structure)\n")
  cat("Python:\n")
  cat("  sklearn.mixture + PCA / factor extraction (approximate)\n")
  cat("  semopy                             -- SEM without native FMM\n")
  cat("Refs: Yung, Y.-F. (1997) 'Finite mixtures in confirmatory factor-analysis\n")
  cat("      models', Psychometrika; Lubke & Muthen (2005) 'Investigating population\n")
  cat("      heterogeneity with factor mixture models', Psychological Methods.\n")
}
