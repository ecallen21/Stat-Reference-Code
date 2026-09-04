# Immortal-time bias (Reference Sec 38.25)
# Native R via survival::tmerge + coxph; Python custom + lifelines.
# Run with:  Rscript immortal_time_bias.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survival::tmerge, coxph        -- build time-varying dataset + Cox fit\n")
  cat("  TrialEmulation                 -- target-trial-emulation for pharmacoepi\n")
  cat("Python:\n")
  cat("  lifelines.CoxTimeVaryingFitter -- time-varying Cox model\n")
  cat("  zepid                          -- pharmaco-epi toolbox with ITB helpers\n")
  cat("  custom                         -- reference implementation\n")
  cat("Refs: Suissa, S. (2008) 'Immortal time bias in pharmacoepidemiology', AJE;\n")
  cat("      Levesque, Hanley, Kezouh, Suissa (2010) 'Problem of immortal time bias\n")
  cat("      in cohort studies', BMJ; Lund, Richardson & Sturmer (2015) 'The active\n")
  cat("      comparator, new user study design in pharmacoepidemiology', Pharm Drug Saf.\n")
}
