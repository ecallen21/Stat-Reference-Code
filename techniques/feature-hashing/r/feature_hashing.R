# Feature hashing / the hashing trick (Reference Sec 41.10)
# Native R via FeatureHashing / text2vec; Python sklearn + custom.
# Run with:  Rscript feature_hashing.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  FeatureHashing::hashed.model.matrix -- signed hashing to fixed dim\n")
  cat("  text2vec::hash_vectorizer            -- text-oriented hashing\n")
  cat("Python:\n")
  cat("  sklearn.feature_extraction.FeatureHasher\n")
  cat("  category_encoders.HashingEncoder\n")
  cat("Refs: Weinberger, Dasgupta, Langford, Smola & Attenberg (2009) 'Feature\n")
  cat("      hashing for large scale multitask learning', ICML.\n")
}
