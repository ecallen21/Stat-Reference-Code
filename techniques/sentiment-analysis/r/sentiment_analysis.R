# Sentiment analysis (Reference §25.7)
# R via tidytext, sentimentr, or syuzhet.
# Run with:  Rscript sentiment_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  tidytext::get_sentiments('bing' | 'afinn' | 'nrc')   -- lexicons\n")
  cat("  sentimentr::sentiment(text)                          -- polarity with negation / amplifiers\n")
  cat("  syuzhet::get_sentiment(text, method='syuzhet' | 'afinn' | 'bing' | 'nrc')\n")
  cat("  vader::get_vader(text)                                -- VADER port for R\n")
  cat("  quanteda.textmodels for supervised sentiment classification.\n")
  cat("Python: nltk.sentiment.vader.SentimentIntensityAnalyzer; TextBlob; transformers pipeline\n")
  cat("        e.g. cardiffnlp/twitter-roberta-base-sentiment-latest\n")
}
