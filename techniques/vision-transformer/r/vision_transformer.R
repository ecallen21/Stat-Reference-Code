# Vision Transformer (Reference §27.x extra)
# R via torch, reticulate + torchvision / timm.
# Run with:  Rscript vision_transformer.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: nn_module with patch embed via nn_conv2d(kernel_size=patch, stride=patch)\n")
  cat("  reticulate + torchvision.models.vit_b_16 (pretrained on ImageNet-1k / 21k)\n")
  cat("  reticulate + timm.create_model('vit_base_patch16_224', pretrained=True)\n")
  cat("ViT variants:\n")
  cat("  * ViT-B/L/H (Dosovitskiy 2021)     -- original\n")
  cat("  * DeiT (Touvron 2020)                -- distillation + strong aug for data efficiency\n")
  cat("  * Swin Transformer (Liu 2021)        -- hierarchical windowed attention\n")
  cat("  * BEiT (Bao 2021), MAE (He 2022)    -- masked-image pretraining\n")
  cat("  * DINOv2 (Oquab 2023)                -- self-supervised general-purpose visual features\n")
  cat("  * SAM (Kirillov 2023)                -- segment anything; ViT backbone + prompt encoder\n")
  cat("  * ConvNeXt / MambaVision              -- competitive convolutional / SSM alternatives\n")
  cat("  * CLIP / SigLIP                       -- image-text contrastive; ViT + text tower\n")
}
