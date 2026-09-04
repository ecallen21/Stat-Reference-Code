# Differential expression analysis (Reference Sec 40.3)
# Native R via limma / DESeq2 / edgeR; Python pydeseq2 + custom.
# Run with:  Rscript differential_expression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  limma::lmFit + eBayes           -- moderated t for microarray / voom(RNA-seq)\n")
  cat("  DESeq2                          -- NB GLM with LFC shrinkage (apeglm/normal/ashr)\n")
  cat("  edgeR                           -- NB GLM + quasi-likelihood F test\n")
  cat("Python:\n")
  cat("  pydeseq2                        -- Python DESeq2 port\n")
  cat("  scanpy.tl.rank_genes_groups     -- single-cell DE\n")
  cat("  diffxpy                         -- multi-model DE for single-cell\n")
  cat("Refs: Ritchie et al. (2015) 'limma powers DE analyses for RNA-seq and microarray',\n")
  cat("      NAR; Love, Huber & Anders (2014) 'Moderated estimation of fold change with\n")
  cat("      DESeq2', Genome Biology.\n")
}
