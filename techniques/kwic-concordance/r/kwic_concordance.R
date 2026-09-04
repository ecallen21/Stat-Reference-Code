# KWIC / concordance (Reference Sec 42.20)
# Native R via quanteda::kwic; Python nltk / textacy + custom.
# Run with:  Rscript kwic_concordance.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quanteda::kwic                  -- KWIC displays w/ window control\n")
  cat("  quanteda::textstat_collocations -- statistical collocations (G^2 / lambda)\n")
  cat("  tm::content_transformer + gregexpr -- ad-hoc concordance\n")
  cat("Python:\n")
  cat("  nltk.Text.concordance            -- NLTK's built-in KWIC\n")
  cat("  textacy.extract.kwic             -- KWIC with token filters\n")
  cat("  spacy + custom                    -- context extraction via linguistic features\n")
  cat("Refs: McEnery & Hardie (2012) Corpus Linguistics: Method, Theory and Practice,\n")
  cat("      Cambridge University Press; Stefanowitsch & Gries (2003) 'Collostructions',\n")
  cat("      Int J Corpus Ling.\n")
}
