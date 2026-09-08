# Product Quantization (Reference Sec 47.84)
# Native R limited (RcppFaiss); Python via faiss / scann.
# Run with:  Rscript product_quantization_pq.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  RcppFaiss                    -- limited FAISS bindings\n")
  cat("  RcppHNSW                     -- HNSW alternative for ANN in R\n")
  cat("Python:\n")
  cat("  faiss                        -- Meta reference: IndexPQ, IVFPQ, OPQ\n")
  cat("  scann                        -- Google's ScaNN with PQ + reranking\n")
  cat("  milvus / weaviate / qdrant   -- production vector DBs (PQ + IVF + HNSW)\n")
  cat("  from-scratch                 -- see product_quantization_pq.py\n")
  cat("Refs: Jegou, Douze & Schmid (2011) IEEE TPAMI 33(1); Ge, He, Ke & Sun\n")
  cat("      (2014) 'Optimized product quantization', IEEE TPAMI 36(4).\n")
}
