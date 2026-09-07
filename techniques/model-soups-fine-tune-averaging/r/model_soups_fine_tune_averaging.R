# Model soups -- weight averaging across fine-tunes (Reference Sec 47.28)
# Deep-learning post-training; Python via torch state_dict averaging.
# Run with:  Rscript model_soups_fine_tune_averaging.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for model soups on deep nets\n")
  cat("Python:\n")
  cat("  Wortsman github release   -- reference model-soups code\n")
  cat("  torch state_dict averaging: {k: mean([m.state_dict()[k] for m in ms])}\n")
  cat("  timm ensemble scripts, mergekit -- LM weight merging framework\n")
  cat("  huggingface_hub load-and-average\n")
  cat("Refs: Wortsman, M. et al. (2022) 'Model soups: averaging weights of\n")
  cat("      multiple fine-tuned models improves accuracy without increasing\n")
  cat("      inference time', ICML; Ilharco, G. et al. (2023) 'Task arithmetic'\n")
  cat("      and 'Editing models with task arithmetic', ICLR.\n")
}
