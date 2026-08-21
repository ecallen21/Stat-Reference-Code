# BLEU / ROUGE / generation eval (Reference §25.x extra)
# R via textcat, xgboost + custom metrics, or Python bridge.
# Run with:  Rscript bleu_rouge_eval.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  NLP + openNLP + tm     -- manual n-gram overlap; no dedicated BLEU package on CRAN\n")
  cat("  reticulate + sacrebleu (Python) or nltk.translate.bleu_score\n")
  cat("  rouge (github.com/pltrdy/rouge; Python) via reticulate\n")
  cat("Metrics:\n")
  cat("  * SacreBLEU — standardised, tokeniser-independent BLEU for MT.\n")
  cat("  * chrF / chrF++ — character-n-gram F; strong for morphologically rich languages.\n")
  cat("  * BERTScore — cosine sim of BERT contextual embeddings; better correlation with humans.\n")
  cat("  * COMET — reference-free neural metric for MT (Unbabel).\n")
  cat("Python: sacrebleu, nltk.translate.bleu_score, rouge_score, bert_score, comet-model.\n")
}
