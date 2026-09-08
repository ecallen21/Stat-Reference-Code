# AdaBoost classifier (Reference Sec 47.69)
# Native R via adabag / fastAdaboost; Python via sklearn.
# Run with:  Rscript adaboost_classifier.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  adabag::boosting          -- AdaBoost.M1 with tree base learners\n")
  cat("  fastAdaboost              -- C++ speed for large n\n")
  cat("  gbm                       -- gradient-boosted trees (successor family)\n")
  cat("Python:\n")
  cat("  sklearn.ensemble.AdaBoostClassifier / AdaBoostRegressor\n")
  cat("  imbalanced-learn.RUSBoostClassifier -- imbalanced-data variant\n")
  cat("  from-scratch                        -- see adaboost_classifier.py\n")
  cat("Refs: Freund & Schapire (1997) JCSS 55(1); Friedman, Hastie & Tibshirani\n")
  cat("      (2000) 'Additive logistic regression', Ann Stat 28(2).\n")
}
