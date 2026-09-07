# Anchor explanations (Reference Sec 47.32)
# Explanation via IF-THEN rules; Python via alibi.explainers.AnchorTabular.
# Run with:  Rscript anchor_explanations.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No direct R port; use ellmer / reticulate to call alibi from Python\n")
  cat("Python:\n")
  cat("  alibi.explainers.AnchorTabular / AnchorText / AnchorImage (Seldon)\n")
  cat("  anchor (github.com/marcotcr/anchor) -- Ribeiro's reference impl\n")
  cat("  interpret / interpret-community -- broader XAI stack, EBM + others\n")
  cat("Refs: Ribeiro, M.T., Singh, S. & Guestrin, C. (2018) 'Anchors: high-\n")
  cat("      precision model-agnostic explanations', AAAI; Molnar, C. (2022)\n")
  cat("      Interpretable Machine Learning ch 5.10.\n")
}
