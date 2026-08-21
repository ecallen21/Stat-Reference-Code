# Named-entity recognition (Reference §25.8)
# R via crfsuite, spacyr, or entity dictionaries.
# Run with:  Rscript named_entity_recognition.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  crfsuite::crf(x=features, y=labels, method='lbfgs')     -- linear-chain CRF\n")
  cat("  spacyr::spacy_parse(text, entity=TRUE)                   -- spaCy models via reticulate\n")
  cat("  udpipe::udpipe_annotate(model, x)                        -- POS + NER (some models)\n")
  cat("  quanteda::tokens_lookup(tokens, dictionary)               -- dictionary-based NER\n")
  cat("Python: spacy nlp.pipe (statistical / transformer NER),\n")
  cat("        flair.models.SequenceTagger (BiLSTM-CRF),\n")
  cat("        huggingface transformers pipeline('ner') with BERT-NER models.\n")
}
