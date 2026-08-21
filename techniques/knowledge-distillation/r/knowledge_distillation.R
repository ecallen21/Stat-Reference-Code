# Knowledge distillation (Reference §27.x extra)
# R via torch or reticulate + Python.
# Run with:  Rscript knowledge_distillation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: nn_kl_div_loss between softmax(z_teacher/T) and log_softmax(z_student/T)\n")
  cat("Python:\n")
  cat("  torch.nn.functional.kl_div + softmax with temperature scaling\n")
  cat("  huggingface transformers.DistilBertModel / DistilRoBERTa\n")
  cat("  timm --distiller or MEAL / KD in ffcv\n")
  cat("Variants:\n")
  cat("  * Feature distillation: match intermediate features (FitNet, RKD).\n")
  cat("  * Response distillation: match softmax outputs (Hinton 2015).\n")
  cat("  * Self-distillation: student and teacher are the same size (Zhang 2019, BAN).\n")
  cat("  * Data-free distillation: generate synthetic samples via GAN.\n")
  cat("  * LLM distillation: MiniLM, DistilGPT2, Alpaca (SFT on GPT-4 outputs).\n")
  cat("Applications: mobile / edge deployment; compressing large transformer models 4-40x.\n")
}
