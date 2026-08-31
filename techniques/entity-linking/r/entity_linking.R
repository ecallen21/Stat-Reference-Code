# Entity linking (Reference §25.x extra)
# R via reticulate + Python.
# Run with:  Rscript entity_linking.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  BLINK (Wu 2020)          -- Facebook bi-encoder + cross-encoder EL over Wikipedia\n")
  cat("  GENRE (Cao 2021)         -- generative EL by autoregressive Wikipedia-title decoding\n")
  cat("  ReFinED (Ayoola 2022)    -- fast BERT-based EL over Wikidata\n")
  cat("  spaCy EntityLinker        -- pipeline component; requires knowledge base\n")
  cat("  BLINK-Docker             -- pre-built inference container\n")
  cat("Knowledge bases: Wikidata, Wikipedia, DBpedia, YAGO, UMLS (medical), ChEBI (chemistry).\n")
  cat("Related: Wikification, Named-Entity Disambiguation (NED), Coreference to knowledge base.\n")
}
