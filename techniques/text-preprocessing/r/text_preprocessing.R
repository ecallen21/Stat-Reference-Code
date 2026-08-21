# Text preprocessing basics (Reference §25.1)
# R via tokenizers, tm, SnowballC, or the tidytext ecosystem.
# Run with:  Rscript text_preprocessing.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  tokenizers::tokenize_words(txt)                      -- fast word tokeniser\n")
  cat("  tm::stopwords('en')  or stopwords::stopwords         -- stopword lists\n")
  cat("  SnowballC::wordStem(w, language='en')                -- Porter / Snowball stemmer\n")
  cat("  textstem::lemmatize_words(w)                         -- rule-based lemmatiser\n")
  cat("  udpipe::udpipe_annotate(model, x)                    -- statistical lemma + POS + parse\n")
  cat("  tidytext::unnest_tokens(df, word, text)              -- tidy pipeline\n")
  cat("  quanteda::tokens(corpus) + tokens_wordstem           -- production-grade\n")
  cat("Python: nltk, spacy, stanza, huggingface tokenizers (BPE / WordPiece / SentencePiece).\n")
}
