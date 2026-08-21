# Quantisation + pruning (Reference §27.x extra)
# R via torch or reticulate + Python.
# Run with:  Rscript quantization_pruning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch: torch::.quantize_per_tensor / .quantize_per_channel\n")
  cat("  torch::nn_utils_prune_l1_unstructured / global_unstructured\n")
  cat("Python:\n")
  cat("  torch.quantization: post-training static / dynamic / QAT (quantisation-aware training)\n")
  cat("  torch.nn.utils.prune: {l1, ln, random}_unstructured / structured\n")
  cat("  bitsandbytes: int8 / int4 quantised linear layers for LLM inference\n")
  cat("  llm-int8 (Dettmers 2022)     -- weight-only + mixed-precision handling of outliers\n")
  cat("  GPTQ (Frantar 2023)          -- group-wise quantisation via approximate second-order\n")
  cat("  AWQ (Lin 2023)               -- activation-aware weight quantisation\n")
  cat("  QLoRA (Dettmers 2023)        -- 4-bit base + LoRA adapters; cheap fine-tuning of LLMs\n")
  cat("Hardware:\n")
  cat("  * int8 / int4 -- Ampere / Hopper Tensor Cores, Apple ANE, Qualcomm Hexagon.\n")
  cat("  * 2:4 structured sparsity   -- NVIDIA Ampere+, 1.5x throughput at same memory.\n")
  cat("  * FP8 (E4M3, E5M2)          -- H100 training + inference; new production default.\n")
}
