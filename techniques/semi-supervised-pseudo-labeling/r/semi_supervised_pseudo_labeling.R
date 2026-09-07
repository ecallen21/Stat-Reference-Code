# Semi-supervised pseudo-labelling (Reference Sec 47.39)
# Native R via mlr3verse / SSL; Python via scikit-learn SelfTrainingClassifier.
# Run with:  Rscript semi_supervised_pseudo_labeling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  RSSL                   -- semi-supervised learning framework\n")
  cat("  mlr3verse (custom pipe with pseudo-label task)\n")
  cat("  SSL                    -- basic self-training and label propagation\n")
  cat("Python:\n")
  cat("  sklearn.semi_supervised.SelfTrainingClassifier -- pseudo-labelling wrapper\n")
  cat("  sklearn.semi_supervised.LabelPropagation / LabelSpreading\n")
  cat("  scikit-uplift / snorkel -- weak-supervision label engineering\n")
  cat("Refs: Lee, D.-H. (2013) 'Pseudo-label: the simple and efficient\n")
  cat("      semi-supervised learning method for deep neural networks', ICML\n")
  cat("      Workshop; Sohn, K. et al. (2020) 'FixMatch: simplifying semi-\n")
  cat("      supervised learning with consistency and confidence', NeurIPS.\n")
}
