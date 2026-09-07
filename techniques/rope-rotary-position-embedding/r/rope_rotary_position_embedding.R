# RoPE -- rotary position embedding (Reference Sec 47.36)
# LLM architecture element; Python via rotary-embedding-torch / xFormers.
# Run with:  Rscript rope_rotary_position_embedding.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for RoPE\n")
  cat("Python:\n")
  cat("  rotary-embedding-torch      -- Phil Wang reference implementation\n")
  cat("  transformers (HF) applies RoPE inside Llama / Mistral / Qwen / Gemma\n")
  cat("  xFormers RoPE kernels        -- Meta AI optimised attention\n")
  cat("  jax + jax.numpy              -- from-scratch RoPE tensor rotation\n")
  cat("Refs: Su, J. et al. (2021) 'RoFormer: enhanced transformer with rotary\n")
  cat("      position embedding', Neurocomputing (2024); Kexuefm blog for RoPE\n")
  cat("      derivation and NTK-aware / YaRN extrapolation extensions.\n")
}
