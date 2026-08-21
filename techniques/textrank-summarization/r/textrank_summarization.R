# TextRank extractive summarisation (Reference §25.12)
# R via textrank or LSAfun.
# Run with:  Rscript textrank_summarization.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  textrank::textrank_sentences(data, terminology)   -- Mihalcea-Tarau TextRank\n")
  cat("  textrank::textrank_keywords(x)                     -- keyword extraction sibling\n")
  cat("  LexRankR::lexRank(text, n)                         -- LexRank (Erkan-Radev 2004; similar idea)\n")
  cat("  LSAfun / lsa                                       -- LSA-based summarisation\n")
  cat("Python: sumy TextRankSummarizer / LexRankSummarizer, summa.summarizer.summarize,\n")
  cat("        gensim.summarization (deprecated in gensim 4.x — use sumy or summa).\n")
  cat("Abstractive: transformer models (BART-large-cnn, T5, Pegasus) via huggingface pipeline.\n")
}
