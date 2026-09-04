# Text preprocessing pipeline (Reference Sec 42.11)
# Native R via quanteda / tidytext; Python nltk / spacy + custom.
# Run with:  Rscript text_preprocessing_pipeline.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quanteda::tokens + tokens_tolower + tokens_remove + tokens_wordstem\n")
  cat("  tidytext::unnest_tokens + get_stopwords\n")
  cat("  tm::tm_map (removePunctuation, tolower, removeWords, stemDocument)\n")
  cat("  SnowballC                       -- Porter / Snowball stemmer\n")
  cat("Python:\n")
  cat("  nltk::word_tokenize + PorterStemmer / SnowballStemmer + WordNetLemmatizer\n")
  cat("  spacy (nlp.lemma_) -- production lemmatiser + POS\n")
  cat("  sklearn.feature_extraction.text.CountVectorizer (basic tokenisation)\n")
  cat("Refs: Manning, Raghavan & Schutze (2008) Introduction to Information Retrieval,\n")
  cat("      Cambridge University Press, Ch 2 (free online); Welbers, van Atteveldt &\n")
  cat("      Benoit (2017) 'Text analysis in R', Comm Methods Meas.\n")
}
