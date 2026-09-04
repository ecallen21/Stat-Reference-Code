# Agreement beyond Cohen's kappa (Reference Sec 38.19)
# Native R via irrCAC / irr; Python custom + krippendorff.
# Run with:  Rscript agreement_beyond_kappa.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  irrCAC::gwet.ac1, gwet.ac2      -- paradox-resistant chance corrections\n")
  cat("  irr::kripp.alpha                -- Krippendorff alpha (any level, any raters)\n")
  cat("  psych::cohen.kappa, ICC         -- classical kappa + intraclass correlation\n")
  cat("Python:\n")
  cat("  krippendorff                    -- Krippendorff alpha\n")
  cat("  sklearn.metrics.cohen_kappa_score\n")
  cat("  custom                          -- PABAK, Gwet AC1, alpha (nominal)\n")
  cat("Refs: Gwet, K.L. (2014) Handbook of Inter-Rater Reliability, 4th ed.;\n")
  cat("      Krippendorff, K. (2019) Content Analysis, 4th ed., SAGE;\n")
  cat("      Byrt, Bishop & Carlin (1993) 'Bias, prevalence, and kappa', J Clin Epi.\n")
}
