# Time-window bias (Reference Sec 43.15)
# Native R via EHR + acs; Python custom + zepid.
# Run with:  Rscript time_window_bias.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  EHR                              -- exposure-algorithm validation\n")
  cat("  acs                              -- adherence calculation from claims\n")
  cat("Python:\n")
  cat("  custom pandas                    -- PDC/MPR calculation from prescription fills\n")
  cat("  zepid                            -- causal-inference / pharmacoepi utilities\n")
  cat("Refs: Suissa & Dell'Aniello (2012) 'Time-window bias in case-control studies:\n")
  cat("      statins and lung cancer', Epidemiology; Schneeweiss & Avorn (2005) 'A\n")
  cat("      review of uses of health care utilization databases for epidemiologic\n")
  cat("      research on therapeutics', J Clin Epi.\n")
}
