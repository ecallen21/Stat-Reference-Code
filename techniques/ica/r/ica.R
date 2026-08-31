# Independent Component Analysis (Reference Sec 25.1)
# Native R via fastICA; Python via sklearn.
# Run with:  Rscript ica.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fastICA                     -- FastICA reference (Hyvarinen)\n")
  cat("  ica                          -- multiple ICA variants (Infomax, JADE)\n")
  cat("  Rica                         -- reproducible ICA + bootstrap CIs\n")
  cat("Python:\n")
  cat("  sklearn.decomposition.FastICA  -- reference FastICA implementation\n")
  cat("  picard                        -- Preconditioned ICA for Real Data (fast)\n")
  cat("  MNE-Python                     -- ICA for EEG / MEG source separation\n")
  cat("Refs: Hyvarinen, A. & Oja, E. (2000) 'Independent Component Analysis:\n")
  cat("      Algorithms and Applications', Neural Networks 13(4).\n")
}
