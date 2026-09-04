# Batch-effect correction (Reference Sec 40.11, 40.14)
# Native R via sva::ComBat; Python neuroCombat + custom.
# Run with:  Rscript batch_effect_combat.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  sva::ComBat, ComBat_seq         -- empirical-Bayes batch correction\n")
  cat("  sva::sva                        -- surrogate variable analysis (unknown batch)\n")
  cat("  limma::removeBatchEffect        -- linear-model residualisation\n")
  cat("  batchelor::fastMNN / mnnCorrect -- single-cell batch integration\n")
  cat("  harmony (RunHarmony)            -- fast single-cell integration\n")
  cat("Python:\n")
  cat("  neuroCombat                     -- Python ComBat for neuroimaging/omics\n")
  cat("  scanpy.pp.combat + bbknn        -- single-cell batch correction\n")
  cat("  harmonypy (run_harmony)         -- Harmony port\n")
  cat("  scvi-tools                      -- deep-learning batch integration\n")
  cat("Refs: Johnson, Li & Rabinovic (2007) 'Adjusting batch effects in microarray\n")
  cat("      expression data using empirical Bayes methods', Biostatistics;\n")
  cat("      Korsunsky et al. (2019) 'Harmony', Nat Methods.\n")
}
