# Orthogonal Matching Pursuit (Reference Sec 47.123)
# Native R via custom (or glmnet); Python via sklearn.
# Run with:  Rscript orthogonal_matching_pursuit.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class OMP -- roll your own with lm / lars ordering\n")
  cat("  glmnet                       -- coord descent lasso baseline\n")
  cat("  BeSS                          -- best-subset selection alternative\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.OrthogonalMatchingPursuit / OrthogonalMatchingPursuitCV\n")
  cat("  scipy.linalg.lstsq            -- refit step\n")
  cat("  from-scratch                  -- see orthogonal_matching_pursuit.py\n")
  cat("Refs: Pati, Rezaiifar & Krishnaprasad (1993) Asilomar; Tropp (2004) IEEE\n")
  cat("      TIT 50(10).\n")
}
