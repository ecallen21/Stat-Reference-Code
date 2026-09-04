# Weighted gene co-expression network analysis (Reference Sec 40.6)
# Native R via WGCNA; Python PyWGCNA + custom.
# Run with:  Rscript wgcna_coexpression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  WGCNA::blockwiseModules         -- full pipeline (soft thresholding + TOM + modules)\n")
  cat("  WGCNA::moduleEigengenes         -- module eigengene extraction\n")
  cat("  hdWGCNA                         -- single-cell WGCNA\n")
  cat("Python:\n")
  cat("  PyWGCNA                         -- Python port of WGCNA\n")
  cat("  hdWGCNA (via rpy2)              -- single-cell WGCNA\n")
  cat("  custom                          -- soft threshold + TOM + hclust\n")
  cat("Refs: Langfelder, P. & Horvath, S. (2008) 'WGCNA: an R package for weighted\n")
  cat("      correlation network analysis', BMC Bioinformatics; Zhang, B. & Horvath, S.\n")
  cat("      (2005) 'A general framework for weighted gene co-expression network\n")
  cat("      analysis', Stat Appl Genet Mol Biol.\n")
}
