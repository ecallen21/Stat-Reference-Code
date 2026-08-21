# n-gram language model + smoothing (Reference §25.x extra)
# R via tm, quanteda, or ngramr.
# Run with:  Rscript ngram_language_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quanteda::tokens_ngrams(x, n=2:3)                          -- fast n-gram tokens\n")
  cat("  tm::TermDocumentMatrix + tm::NGramTokenizer\n")
  cat("  text2vec::create_dtm + word ngrams\n")
  cat("  tidytext::unnest_tokens(df, ngram, text, token='ngrams', n=3)\n")
  cat("Python: nltk.lm.MLE / Lidstone / KneserNeyInterpolated / WittenBellInterpolated,\n")
  cat("        kenlm (production KN-smoothed LM), sentencepiece + subword LMs.\n")
  cat("Modern replacement: transformer causal LM (GPT-style) with perplexity 10-30x lower.\n")
}
