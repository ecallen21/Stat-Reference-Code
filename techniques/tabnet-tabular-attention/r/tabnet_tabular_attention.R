# TabNet tabular attention (Reference Sec 47.112)
# Deep learning; Python via pytorch-tabnet.
# Run with:  Rscript tabnet_tabular_attention.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class TabNet in R -- reticulate wraps pytorch-tabnet\n")
  cat("  torch R                        -- build custom TabNet manually\n")
  cat("Python:\n")
  cat("  pytorch-tabnet                 -- reference implementation with pretraining\n")
  cat("  keras-tabnet                   -- Keras port\n")
  cat("  tab-transformer / ft-transformer -- transformer alternatives for tabular\n")
  cat("  xgboost / lightgbm / catboost  -- typical baselines to beat\n")
  cat("  from-scratch                    -- see tabnet_tabular_attention.py\n")
  cat("Refs: Arik & Pfister (2021) 'TabNet: Attentive interpretable tabular\n")
  cat("      learning', AAAI; Martins & Astudillo (2016) 'From softmax to\n")
  cat("      sparsemax', ICML.\n")
}
