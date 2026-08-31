# Abstractive summarisation (Reference §25.x extra)
# R via reticulate + Python (huggingface transformers).
# Run with:  Rscript abstractive_summarization.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R for abstractive summarisation.\n")
  cat("Python:\n")
  cat("  transformers.pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')\n")
  cat("  transformers 'facebook/bart-large-cnn'   -- CNN/DailyMail SOTA-ish\n")
  cat("  transformers 'google/pegasus-xsum'       -- Pegasus (Zhang 2020)\n")
  cat("  transformers 'google/flan-t5-large'      -- T5 fine-tuned on many tasks\n")
  cat("  sumy (extractive baselines: LexRank, LSA, KL-sum)\n")
  cat("Datasets: CNN/DailyMail, XSum, PubMed, arXiv, MediaSum.\n")
  cat("Evaluation: ROUGE + BERTScore; hallucination detection (SummaC, QAGS);\n")
  cat("            human evaluation for faithfulness + coherence.\n")
}
