# Word-sense disambiguation (Reference §25.x extra)
# R via wordnet or reticulate + Python nltk / pywsd / transformers.
# Run with:  Rscript word_sense_disambiguation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  wordnet, WordNetR                              -- interfaces to Princeton WordNet\n")
  cat("  reticulate + nltk.wsd.lesk / pywsd / babelnet-py\n")
  cat("Modern WSD:\n")
  cat("  GlossBERT (Huang 2019)   -- BERT sentence-pair classifier over (context, gloss)\n")
  cat("  BEM (Blevins-Zettlemoyer) -- bi-encoder mapping context and gloss into same space\n")
  cat("  ConSeC (Barba 2021)       -- current SOTA on ALL-WSD benchmarks\n")
  cat("  Zero-shot via GPT-family: prompt with sense definitions; classify.\n")
}
