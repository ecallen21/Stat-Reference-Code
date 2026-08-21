# Coreference resolution (Reference §25.x extra)
# R via reticulate + Python NeuralCoref / spaCy / AllenNLP; no native R package.
# Run with:  Rscript coreference_resolution.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No mature native R packages for coreference; use reticulate + Python:\n")
  cat("    spacy nlp.add_pipe('coreferee') / 'crosslingual-coreference'\n")
  cat("    allennlp.predictors.CorefPredictor\n")
  cat("    fastcoref (Otmazgin 2022) — fast SpanBERT-based coreference\n")
  cat("    huggingface transformers — LingMess, s2e-coref, etc.\n")
  cat("Classical:\n")
  cat("  Hobbs 1978 rule-based tree traversal\n")
  cat("  Lappin-Leass 1994 salience-based pronoun resolution\n")
  cat("  Soon-Ng-Lim 2001 mention-pair classifier\n")
  cat("  Bengtson-Roth 2008 richer features + averaged perceptron\n")
  cat("  Ng-Cardie 2002 entity-mention model\n")
  cat("Modern SOTA: end-to-end span-based coreference (Lee 2017; ")
  cat("s2e-coref 2021; LingMess 2022).\n")
}
