# Collocation statistics (Reference Sec 42.12)
# Native R via quanteda::textstat_collocations; Python nltk + custom.
# Run with:  Rscript collocation_pmi.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quanteda::textstat_collocations -- lambda + G^2 collocations, ngram sizes\n")
  cat("  tidytext::unnest_tokens(ngrams=2) -- bigram extraction\n")
  cat("Python:\n")
  cat("  nltk.collocations (BigramAssocMeasures, TrigramAssocMeasures)\n")
  cat("  gensim::Phrases                  -- automatic collocation detection\n")
  cat("  sklearn.feature_extraction.text.CountVectorizer(ngram_range)\n")
  cat("Refs: Dunning (1993) 'Accurate methods for the statistics of surprise and\n")
  cat("      coincidence', Comput Ling; Manning & Schutze (1999) Foundations of\n")
  cat("      Statistical NLP, MIT Press, Ch 5.\n")
}
