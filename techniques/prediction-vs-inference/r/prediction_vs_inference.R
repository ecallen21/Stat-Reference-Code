# Prediction vs inference (Reference Sec 39.1)
# Conceptual + workflow contrast; Python demonstrates numerically.
# Run with:  Rscript prediction_vs_inference.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rms                            -- Harrell's inference-first modelling toolbox\n")
  cat("  caret / tidymodels             -- prediction-first workflows\n")
  cat("  MASS::stepAIC                  -- stepwise (AVOID for prediction, per Harrell)\n")
  cat("Python:\n")
  cat("  statsmodels                    -- OLS/Logit with SE, CI, hypothesis tests\n")
  cat("  sklearn                        -- Pipeline + cross_val_score for prediction\n")
  cat("Refs: Shmueli, G. (2010) 'To explain or to predict?', Statistical Science;\n")
  cat("      Steyerberg, E.W. (2019) Clinical Prediction Models, 2nd ed., Springer;\n")
  cat("      Harrell, F.E. (2015) Regression Modeling Strategies, 2nd ed., Springer.\n")
}
