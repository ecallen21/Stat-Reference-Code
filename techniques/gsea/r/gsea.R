# Gene set enrichment analysis (Reference Sec 40.4, 40.18)
# Native R via fgsea + clusterProfiler; Python gseapy + custom.
# Run with:  Rscript gsea.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fgsea::fgsea                    -- fast weighted KS enrichment score + p\n")
  cat("  clusterProfiler::gseGO/gseKEGG  -- GO / KEGG GSEA + visualisation\n")
  cat("  GSVA                            -- single-sample GSEA (ssGSEA)\n")
  cat("  topGO, enrichR                  -- ORA + GO topology methods\n")
  cat("Python:\n")
  cat("  gseapy (gsea, prerank, enrichr) -- Python GSEA reference\n")
  cat("  goatools                        -- gene-ontology enrichment\n")
  cat("  gProfiler API                   -- online multi-database enrichment\n")
  cat("Refs: Subramanian et al. (2005) 'Gene set enrichment analysis', PNAS;\n")
  cat("      Korotkevich, Sukhov & Sergushichev (2019) 'Fast gene set enrichment\n")
  cat("      analysis', bioRxiv (fgsea).\n")
}
