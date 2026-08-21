# Masked Language Modeling (Reference §25.x extra)
# R via torch or reticulate + huggingface transformers.
# Run with:  Rscript masked_language_modeling.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: nn_module encoder + linear head + cross-entropy on masked positions\n")
  cat("  reticulate + transformers pipeline('fill-mask', model='bert-base-uncased')\n")
  cat("  text::textEmbed / textFill                     -- HuggingFace via reticulate wrapper\n")
  cat("Variants:\n")
  cat("  MLM (BERT)  Masked-LM + Next-Sentence-Prediction pretraining\n")
  cat("  RoBERTa    Dynamic masking; drops NSP; longer pretraining\n")
  cat("  ELECTRA     Replaced-token detection (discriminative) instead of generative MLM\n")
  cat("  DeBERTa     Disentangled attention + relative positional encoding\n")
  cat("  T5 / BART   Span-masking + seq2seq (span-corruption) rather than single-token MLM\n")
  cat("  ModernBERT / GTE / mxbai-embed  Recent encoders trained with MLM + downstream contrastive.\n")
}
