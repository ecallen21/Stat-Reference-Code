# MinHash + LSH (Reference Sec 47.76)
# Native R via LSHR / textreuse; Python via datasketch.
# Run with:  Rscript min_hash_lsh.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  LSHR::LSH.minhash          -- MinHash + banded LSH\n")
  cat("  textreuse::minhash_generator + lsh -- reference for text dedup\n")
  cat("  RcppMinHash                -- C++-fast MinHash signatures\n")
  cat("Python:\n")
  cat("  datasketch.MinHash / MinHashLSH  -- production-quality LSH\n")
  cat("  scikit-learn's LSHForest         -- angular LSH (deprecated in newer versions)\n")
  cat("  faiss + hashing indices          -- billion-scale ANN\n")
  cat("  from-scratch                     -- see min_hash_lsh.py\n")
  cat("Refs: Broder (1997) 'On the resemblance and containment of documents',\n")
  cat("      CCS; Indyk & Motwani (1998) STOC.\n")
}
