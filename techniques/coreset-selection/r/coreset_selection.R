# Coreset selection (Reference Sec 47.44)
# Native R support is limited; Python has richer implementations.
# Run with:  Rscript coreset_selection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ClusterR::MiniBatchKmeans   -- mini-batch is a poor-man's coreset\n")
  cat("  onlinePCA                   -- streaming PCA on merged coresets\n")
  cat("  BigVAR                      -- coreset ideas for VAR estimation\n")
  cat("Python:\n")
  cat("  submodlib                   -- Feldman-Langberg, k-means++, GraphCut, etc\n")
  cat("  kmc2                        -- k-MC^2 markov-chain k-means++\n")
  cat("  from-scratch                -- see coreset_selection.py\n")
  cat("Refs: Feldman & Langberg (2011) STOC; Mirzasoleiman, Bilmes & Leskovec\n")
  cat("      (2020) 'Coresets for data-efficient training of ML models', ICML.\n")
}
