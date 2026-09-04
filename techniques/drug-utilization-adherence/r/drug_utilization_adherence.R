# Drug utilization / adherence (Reference Sec 43.7)
# Native R via AdhereR; Python custom + pandas.
# Run with:  Rscript drug_utilization_adherence.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  AdhereR::CMA (CMA1-CMA9)         -- continuous multiple-interval measures\n")
  cat("  AdhereR::CMA_per_episode         -- persistence + episodes of care\n")
  cat("  survey                            -- population estimates + design effects\n")
  cat("Python:\n")
  cat("  pandas (custom)                   -- PDC / MPR / persistence from fills\n")
  cat("  lifelines                         -- time-to-discontinuation survival\n")
  cat("Refs: WHO CC for Drug Statistics Methodology (2024) 'Guidelines for ATC\n")
  cat("      Classification and DDD Assignment'; Andrade et al. (2006) 'Methods for\n")
  cat("      evaluation of medication adherence and persistence using automated\n")
  cat("      databases', Pharmacoepi Drug Saf.\n")
}
