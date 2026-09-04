# New-user, active-comparator design (Reference Sec 43.14)
# Native R via CohortMethod / MatchIt / WeightIt; Python zepid + custom.
# Run with:  Rscript new_user_active_comparator.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  CohortMethod (OHDSI: createStudyPopulation, washout + new-user criteria)\n")
  cat("  MatchIt, WeightIt                -- PS matching / IPTW post-restriction\n")
  cat("  cobalt                            -- balance diagnostics after ACNU\n")
  cat("Python:\n")
  cat("  pandas (custom)                   -- new-user identification via first fill\n")
  cat("  zepid                             -- IPTW + G-formula for ACNU cohorts\n")
  cat("  OHDSI tools via REST API\n")
  cat("Refs: Lund, Richardson & Sturmer (2015) 'The active comparator, new user study\n")
  cat("      design in pharmacoepidemiology', Curr Epi Rep; Ray (2003) 'Evaluating\n")
  cat("      medication effects outside of clinical trials: new-user designs', AJE.\n")
}
