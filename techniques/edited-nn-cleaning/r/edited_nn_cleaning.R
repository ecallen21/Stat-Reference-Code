# Edited Nearest Neighbours (Reference Sec 47.250)
# Native R via unbalanced / UBL / themis; Python via imbalanced-learn / from-scratch.
# Run with:  Rscript edited_nn_cleaning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  unbalanced::ubENN         -- Wilson's edited NN\n")
  cat("  UBL::ENNClassif           -- edited NN in unified framework\n")
  cat("  themis::step_nearmiss     -- tidymodels-friendly under-samplers\n")
  cat("Python:\n")
  cat("  imbalanced-learn.under_sampling.EditedNearestNeighbours\n")
  cat("  imbalanced-learn.under_sampling.AllKNN (repeated ENN)\n")
  cat("  imbalanced-learn.combine.SMOTEENN\n")
  cat("  From-scratch (see edited_nn_cleaning.py)\n")
  cat("Refs: Wilson, D.L. (1972) 'Asymptotic Properties of Nearest Neighbor\n")
  cat("      Rules Using Edited Data', IEEE Trans SMC 2(3), 408-421.\n")
}
