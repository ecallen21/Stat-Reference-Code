# Explainable boosting machine (EBM) (Reference Sec 47.31)
# Native R via interpret; Python via interpret / interpret-community.
# Run with:  Rscript explainable_boosting_machine.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  interpret         -- Microsoft R port of EBM (via reticulate)\n")
  cat("  mboost            -- component-wise boosting (GAM-like fits)\n")
  cat("  gbm / xgboost + shapley -- boosted trees + SHAP for post-hoc interp\n")
  cat("Python:\n")
  cat("  interpret / interpret-community -- Microsoft reference EBM (Caruana)\n")
  cat("  scikit-learn HistGradientBoosting with per-feature graphs\n")
  cat("  pyGAM             -- alternative GAM library\n")
  cat("Refs: Nori, H. et al. (2019) 'InterpretML: A unified framework for machine\n")
  cat("      learning interpretability', arXiv:1909.09223; Lou, Y. et al. (2013)\n")
  cat("      'Accurate intelligible models with pairwise interactions', KDD;\n")
  cat("      Hastie & Tibshirani (1990) Generalized Additive Models, Chapman.\n")
}
