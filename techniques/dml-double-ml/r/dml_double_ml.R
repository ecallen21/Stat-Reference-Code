# Double / Debiased ML (Reference Sec 15.25)
# Native R via DoubleML; Python DoubleML / econml.
# Run with:  Rscript dml_double_ml.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  DoubleML::DoubleMLPLR / IRM / IIVM -- Chernozhukov canonical Python-parity port\n")
  cat("  grf::causal_forest                 -- alternative causal ML with honest splits\n")
  cat("  hdm                                 -- high-dim treatment effects (Belloni)\n")
  cat("Python:\n")
  cat("  DoubleML                            -- reference implementation\n")
  cat("  econml.dml (DML, SparseLinearDML, ForestDML)\n")
  cat("  causalml                            -- meta-learner family\n")
  cat("Refs: Chernozhukov et al. (2018) 'Double/debiased machine learning for\n")
  cat("      treatment and structural parameters', Econometrics Journal.\n")
}
