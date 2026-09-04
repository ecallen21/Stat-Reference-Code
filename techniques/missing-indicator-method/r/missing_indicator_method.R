# Missing indicator method (Reference Sec 41.12)
# Native R via mice / recipes; Python sklearn.impute + custom.
# Run with:  Rscript missing_indicator_method.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  recipes::step_indicate_na       -- automatic missing-indicator creation\n")
  cat("  mice::ampute + custom            -- imputation + optional indicators\n")
  cat("  hmisc::describe                  -- diagnostic before deciding\n")
  cat("Python:\n")
  cat("  sklearn.impute.MissingIndicator + SimpleImputer\n")
  cat("  pandas.DataFrame.isna            -- indicator construction\n")
  cat("Refs: Groenwold, White, Donders, Carpenter, Altman & Moons (2012) 'Missing\n")
  cat("      covariate data in clinical research: when and when not to use the missing\n")
  cat("      indicator method for analysis', CMAJ; Knol et al. (2010) 'Unpredictable\n")
  cat("      bias when using the missing indicator method', J Clin Epi.\n")
}
