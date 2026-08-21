# chrF + BERTScore (Reference §25.x extra)
# R via reticulate + Python bert_score / sacrebleu, no strong native R packages.
# Run with:  Rscript bertscore_chrf_metrics.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No mature native R packages; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  sacrebleu.corpus_chrf(sys, refs, word_order=2)  -- SacreBLEU's chrF++\n")
  cat("  bert_score.score(cands, refs, model_type='microsoft/deberta-xlarge-mnli')\n")
  cat("  unbabel-comet: COMET-22 (reference-based) and COMET-KIWI (reference-free) for MT.\n")
  cat("  MAUVE: distributional similarity for open-ended generation (Pillutla 2021).\n")
  cat("Recommendations by task:\n")
  cat("  * MT: SacreBLEU + chrF + COMET-22 (or COMET-KIWI for reference-free)\n")
  cat("  * Summarisation: ROUGE + BERTScore + a discriminative human eval\n")
  cat("  * Open-ended: MAUVE + LLM-as-judge + calibrated human study.\n")
}
