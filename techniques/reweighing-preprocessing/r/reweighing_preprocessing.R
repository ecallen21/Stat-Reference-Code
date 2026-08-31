# Reweighing pre-processing (Reference Ch 31 Fairness)
# Native R (weights argument to glm); Python via aif360.
# Run with:  Rscript reweighing_preprocessing.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fairml                      -- reweigh(), sample_weight() helpers\n")
  cat("  fairmodels                  -- pre_processing_reweight()\n")
  cat("  glm(..., weights = w)       -- base R: weights on any learner\n")
  cat("Python:\n")
  cat("  aif360.algorithms.preprocessing.Reweighing  (Kamiran-Calders reference)\n")
  cat("  fairlearn.reductions        -- adjacent reduction approaches\n")
  cat("Refs: Kamiran, F. & Calders, T. (2012) 'Data Preprocessing Techniques for\n")
  cat("      Classification without Discrimination', Knowledge and Information Systems.\n")
}
