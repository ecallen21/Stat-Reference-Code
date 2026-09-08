# HNSW ANN search (Reference Sec 47.85)
# Native R via RcppHNSW; Python via hnswlib / faiss / scann.
# Run with:  Rscript hnsw_ann_search.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  RcppHNSW::hnsw_build / hnsw_search   -- Malkov-Yashunin bindings\n")
  cat("  uwot                                  -- UMAP that internally uses HNSW\n")
  cat("  RcppAnnoy                             -- Spotify Annoy (kd-tree ANN)\n")
  cat("Python:\n")
  cat("  hnswlib               -- Malkov-Yashunin reference implementation\n")
  cat("  faiss (IndexHNSWFlat) -- HNSW as one of many FAISS index options\n")
  cat("  scann                 -- Google ScaNN (SoAR + tree-AH)\n")
  cat("  pgvector              -- Postgres extension with HNSW / IVFFLAT\n")
  cat("  from-scratch fallback -- see hnsw_ann_search.py\n")
  cat("Refs: Malkov & Yashunin (2018/2020) 'Efficient and robust approximate\n")
  cat("      nearest neighbor search using HNSW graphs', IEEE TPAMI 42(4).\n")
}
