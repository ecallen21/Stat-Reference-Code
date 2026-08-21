# POS tagging (Reference §25.x extra)
# R via udpipe, spacyr, or NLP.
# Run with:  Rscript pos_tagging.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  udpipe::udpipe_annotate(model, x, tagger='default')      -- statistical POS + lemma + parse\n")
  cat("  spacyr::spacy_parse(text, pos=TRUE, dependency=FALSE)    -- spaCy via reticulate\n")
  cat("  NLP::Annotator + openNLP::Maxent_POS_Tag_Annotator       -- classical Apache OpenNLP\n")
  cat("  RNLP::pos_tag                                             -- lightweight\n")
  cat("Python: nltk.pos_tag (Penn Treebank tags), spacy nlp.pipe, stanza pipeline, flair.\n")
  cat("Modern: BiLSTM-CRF (flair) or transformer + linear head (BERT + POS layer).\n")
}
