# Transfer learning: feature extraction + fine-tuning (Reference §27.x extra)
# R via torch or keras3.
# Run with:  Rscript transfer_learning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torchvision::model_resnet50(pretrained=TRUE)  +  set param$requires_grad_(FALSE)\n")
  cat("  keras3::application_resnet50(include_top=FALSE)  +  freeze_weights()\n")
  cat("  torch::optim_adam(head_params, lr=1e-3)         -- head-only optimiser\n")
  cat("  torch::optim_adam(all_params, lr=1e-5)          -- fine-tune optimiser at small LR\n")
  cat("Standard recipes:\n")
  cat("  * IMAGE: ImageNet-pretrained ResNet / ViT + custom head.\n")
  cat("  * TEXT:  BERT / RoBERTa + linear classifier head; discriminative LR per layer.\n")
  cat("  * AUDIO: wav2vec2 / HuBERT + linear head; freeze feature encoder, unfreeze context.\n")
  cat("  * ULMFiT (Howard-Ruder 2018): gradual unfreezing + slanted triangular LRs for NLP.\n")
  cat("  * LoRA / adapters (Hu 2021): low-rank fine-tuning; near-frozen backbone.\n")
  cat("Python: torchvision.models, timm, huggingface transformers .from_pretrained + fine-tune.\n")
}
