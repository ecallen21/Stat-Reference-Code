# Safe Screening for LASSO (Reference Sec 47.336)
# Native R via glmnet (strong rule) / gglasso; Python via celer / from-scratch.
# Run with:  Rscript safe_screening_lasso.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  glmnet                    -- strong rule built into the solver\n")
  cat("  gglasso                   -- with sequential strong-rule warm starts\n")
  cat("  biglasso                  -- large-scale LASSO with screening\n")
  cat("Python:\n")
  cat("  celer.Lasso                    -- dynamic safe screening\n")
  cat("  sklearn.linear_model.Lasso    -- strong-rule variant\n")
  cat("  From-scratch (see safe_screening_lasso.py)\n")
  cat("Refs: El Ghaoui, L., Viallon, V. & Rabbani, T. (2010) 'Safe feature\n")
  cat("      elimination for the LASSO and sparse supervised learning problems',\n")
  cat("      arXiv:1009.4219;  Fercoq, O., Gramfort, A. & Salmon, J. (2015)\n")
  cat("      'Mind the duality gap: safer rules for the Lasso', ICML.\n")
}
