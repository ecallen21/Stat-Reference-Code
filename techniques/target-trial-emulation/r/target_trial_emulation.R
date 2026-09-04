# Target trial emulation (Reference Sec 43.11)
# Native R via TrialEmulation / CohortMethod (OHDSI); Python zepid + custom.
# Run with:  Rscript target_trial_emulation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  TrialEmulation                    -- Hernan-Robins target-trial framework\n")
  cat("  CohortMethod (OHDSI)              -- new-user, active-comparator on OMOP CDM\n")
  cat("  Cyclops + FeatureExtraction       -- large-scale regularised regression\n")
  cat("Python:\n")
  cat("  TrialEmulation via rpy2\n")
  cat("  zepid::IPTW / GFormula            -- causal-inference toolkit\n")
  cat("  OHDSI tools via REST API\n")
  cat("Refs: Hernan & Robins (2016) 'Using big data to emulate a target trial when a\n")
  cat("      randomized trial is not available', AJE; Dickerman et al. (2019) 'Avoidable\n")
  cat("      flaws in observational analyses: an application to statins and cancer',\n")
  cat("      Nat Med.\n")
}
