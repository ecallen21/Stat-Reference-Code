# Matthews Correlation Coefficient (Reference Sec 47.253)
# Native R via mltools / yardstick / mccr; Python via sklearn / from-scratch.
# Run with:  Rscript matthews_correlation_coefficient.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mltools::mcc              -- binary + multiclass MCC\n")
  cat("  yardstick::mcc            -- tidymodels metric\n")
  cat("  mccr::mccr                -- dedicated MCC package\n")
  cat("  caret::confusionMatrix    -- reports MCC in extras\n")
  cat("Python:\n")
  cat("  sklearn.metrics.matthews_corrcoef\n")
  cat("  From-scratch numpy (see matthews_correlation_coefficient.py)\n")
  cat("Refs: Matthews, B.W. (1975) 'Comparison of the Predicted and Observed\n")
  cat("      Secondary Structure of T4 Phage Lysozyme', BBA 405(2), 442-451;\n")
  cat("      Chicco, D. & Jurman, G. (2020) 'The advantages of the Matthews\n")
  cat("      correlation coefficient over F1 score and accuracy', BMC Genomics 21.\n")
}
