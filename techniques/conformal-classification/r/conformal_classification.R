# Conformal classification / APS + RAPS (Reference Ch 29 Uncertainty Quantification)
# R via reticulate + Python; native R alternatives are limited.
# Run with:  Rscript conformal_classification.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  conformalClassification    -- basic split-conformal classification\n")
  cat("  crepes (via reticulate)    -- conformal regressors + classifiers, sklearn API\n")
  cat("Python:\n")
  cat("  mapie                      -- inductive / cross / adaptive APS + RAPS\n")
  cat("  puncc                      -- Deel-AI conformal-prediction toolbox\n")
  cat("  torchcp                    -- conformal prediction for pytorch classifiers\n")
  cat("Refs: Romano, Y., Sesia, M. & Candes, E. (2020) 'Classification with valid and\n")
  cat("      adaptive coverage', NeurIPS; Angelopoulos, A. et al. (2021)\n")
  cat("      'Uncertainty sets for image classifiers using conformal prediction (RAPS)', ICLR.\n")
}
