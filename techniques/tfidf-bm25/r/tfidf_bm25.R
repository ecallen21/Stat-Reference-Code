# TF-IDF and BM25 (Reference §25.2)
# R via text2vec, quanteda, or tm.
# Run with:  Rscript tfidf_bm25.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  text2vec::create_dtm(it, vect)  +  text2vec::TfIdf$new() -- fast TF-IDF pipeline\n")
  cat("  quanteda::dfm_tfidf(dfm)                                   -- corpus-first workflow\n")
  cat("  tm::weightTfIdf(dtm)                                       -- classical R text package\n")
  cat("  superml::TfIdfVectorizer / CountVectorizer                 -- sklearn-like API\n")
  cat("  BM25 in R: quanteda.textstats::textstat_simil variants; or roll your own with dfm counts.\n")
  cat("Python: sklearn.feature_extraction.text.TfidfVectorizer; rank_bm25.BM25Okapi.\n")
}
