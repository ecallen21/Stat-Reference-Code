# Latent semantic analysis / indexing (Reference Sec 42.13)
# Native R via lsa / text2vec / irlba; Python sklearn + gensim + custom.
# Run with:  Rscript lsa_latent_semantic.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  lsa::lsa                        -- LSA reference implementation\n")
  cat("  text2vec::LSA                    -- rsparse-backed LSA\n")
  cat("  irlba::irlba                     -- fast truncated SVD\n")
  cat("Python:\n")
  cat("  sklearn.decomposition.TruncatedSVD -- LSA baseline\n")
  cat("  gensim.models.LsiModel           -- Gensim LSA/LSI\n")
  cat("Refs: Deerwester, Dumais, Furnas, Landauer & Harshman (1990) 'Indexing by\n")
  cat("      latent semantic analysis', JASIS; Landauer & Dumais (1997) 'A solution\n")
  cat("      to Plato's problem: the LSA theory', Psychol Review.\n")
}
