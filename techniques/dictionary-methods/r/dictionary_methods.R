# Dictionary-based text scoring (Reference Sec 42.14)
# Native R via quanteda; Python empath + custom.
# Run with:  Rscript dictionary_methods.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quanteda::dictionary + tokens_lookup + dfm_lookup\n")
  cat("  tidytext + get_sentiments        -- pre-built sentiment dictionaries\n")
  cat("  LIWC (commercial)                -- Pennebaker's psychosocial lexicon\n")
  cat("Python:\n")
  cat("  empath                            -- open-source LIWC-like categories\n")
  cat("  liwc (commercial)                 -- LIWC via API\n")
  cat("  nltk / custom                     -- domain-specific dictionaries\n")
  cat("Refs: Pennebaker, Boyd, Jordan & Blackburn (2015) 'The development and\n")
  cat("      psychometric properties of LIWC2015'; Young & Soroka (2012) 'Affective\n")
  cat("      news: the automated coding of sentiment in political texts', Political\n")
  cat("      Communication.\n")
}
