# Contrastive Predictive Coding (Reference Sec 47.99)
# Deep-learning workflow; Python via torch / torchaudio.
# Run with:  Rscript contrastive_predictive_coding.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class CPC in R -- SSL for sequences lives in Python\n")
  cat("Python:\n")
  cat("  torchaudio                    -- wav2vec 2.0 / HuBERT CPC-family models\n")
  cat("  transformers                  -- HF wav2vec2, HuBERT, data2vec\n")
  cat("  cpc-audio (Facebook)          -- reference CPC for audio\n")
  cat("  info-nce-pytorch              -- reusable InfoNCE loss\n")
  cat("  from-scratch                  -- see contrastive_predictive_coding.py\n")
  cat("Refs: van den Oord, Li & Vinyals (2018) arXiv:1807.03748; Baevski, Zhou,\n")
  cat("      Mohamed & Auli (2020) 'wav2vec 2.0', NeurIPS.\n")
}
