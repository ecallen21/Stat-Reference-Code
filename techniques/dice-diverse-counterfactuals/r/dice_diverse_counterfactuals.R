# DiCE - Diverse Counterfactual Explanations (Reference Sec 47.254)
# Native R via iml / counterfactuals; Python via dice-ml / from-scratch.
# Run with:  Rscript dice_diverse_counterfactuals.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  counterfactuals            -- WhatIf / NICE / MOC counterfactuals\n")
  cat("  iml::Predictor + custom search -- interpretable ML toolkit\n")
  cat("  dice via reticulate         -- wrap dice-ml Python from R\n")
  cat("Python:\n")
  cat("  dice-ml (Microsoft, PyPI: dice-ml)\n")
  cat("  alibi.explainers.CounterfactualProto\n")
  cat("  From-scratch (see dice_diverse_counterfactuals.py)\n")
  cat("Refs: Mothilal, R.K., Sharma, A. & Tan, C. (2020) 'Explaining Machine\n")
  cat("      Learning Classifiers Through Diverse Counterfactual Explanations',\n")
  cat("      ACM FAT*, 607-617.\n")
}
