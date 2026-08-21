# CKY / probabilistic CFG parsing (Reference §25.x extra)
# R via NLP + openNLP; production parsing is easier via reticulate + Python.
# Run with:  Rscript syntactic_parsing_cky.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  NLP + openNLP::Parse_Annotator                          -- Apache OpenNLP wrapper\n")
  cat("  udpipe::udpipe_annotate(model, x, parser='default')     -- statistical dependency parser\n")
  cat("  spacyr::spacy_parse(dependency=TRUE)                    -- spaCy dependency parser\n")
  cat("Python:\n")
  cat("  nltk.CFG + nltk.ChartParser / ViterbiParser              -- CFG / PCFG parsers\n")
  cat("  benepar (Kitaev-Klein 2018)                              -- transformer-based constituency parser\n")
  cat("  stanza / spaCy / trankit                                  -- dependency parsers (UD schema)\n")
  cat("  huggingface transformers models for constituency + dependency.\n")
}
