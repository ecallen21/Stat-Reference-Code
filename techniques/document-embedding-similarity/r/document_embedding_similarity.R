# Document embedding + similarity (Reference Sec 42.9)
# Native R via text2vec / quanteda; Python sentence-transformers + custom.
# Run with:  Rscript document_embedding_similarity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  text2vec::doc2vec              -- distributed doc vectors\n")
  cat("  text (via sentence-transformers)\n")
  cat("  quanteda + tfidf                -- TF-IDF baseline\n")
  cat("Python:\n")
  cat("  sentence-transformers            -- SBERT (Reimers-Gurevych 2019)\n")
  cat("  gensim::Doc2Vec\n")
  cat("  sklearn.feature_extraction.text.TfidfVectorizer + cosine\n")
  cat("  rank_bm25                        -- BM25 rankers\n")
  cat("Refs: Reimers & Gurevych (2019) 'Sentence-BERT: sentence embeddings using\n")
  cat("      Siamese BERT-networks', EMNLP; Robertson & Sparck-Jones (1976)\n")
  cat("      'Relevance weighting of search terms', JASIS.\n")
}
