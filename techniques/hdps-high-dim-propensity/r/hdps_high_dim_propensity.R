# High-dimensional propensity scores (Reference Sec 43.4)
# Native R via OHDSI FeatureExtraction / CohortMethod; Python sklearn + custom.
# Run with:  Rscript hdps_high_dim_propensity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  FeatureExtraction + CohortMethod (OHDSI) -- hdPS on OMOP CDM\n")
  cat("  glmnet                                    -- LASSO-based PS with high-dim covariates\n")
  cat("  hdps (custom)                             -- direct Schneeweiss implementation\n")
  cat("Python:\n")
  cat("  sklearn.linear_model (LogisticRegression) -- baseline\n")
  cat("  OHDSI tools via REST API\n")
  cat("Refs: Schneeweiss et al. (2009) 'High-dimensional propensity score adjustment\n")
  cat("      in studies of treatment effects using health care claims data',\n")
  cat("      Epidemiology; Rassen et al. (2011) 'One-to-many propensity score matching',\n")
  cat("      Pharmacoepi Drug Saf.\n")
}
