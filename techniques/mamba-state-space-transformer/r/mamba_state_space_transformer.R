# Mamba / structured state-space models (S4, S5) (Reference Sec 47.22)
# Deep-learning inference technique; no R implementations.
# Run with:  Rscript mamba_state_space_transformer.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No R implementations for Mamba/S4/S5.\n")
  cat("Python:\n")
  cat("  mamba-ssm                -- official Mamba impl (Gu & Dao)\n")
  cat("  state-spaces              -- S4/S5 reference code\n")
  cat("  torch + einops            -- from-scratch SSM in PyTorch\n")
  cat("  Hugging Face transformers -- mamba, mamba2 checkpoints\n")
  cat("  s5-pytorch               -- diagonal-A S5 with parallel scan\n")
  cat("Refs: Gu, A., Goel, K. & Re, C. (2022) 'Efficiently modeling long\n")
  cat("      sequences with structured state spaces', ICLR; Gu, A. & Dao, T.\n")
  cat("      (2023) 'Mamba: linear-time sequence modeling with selective state\n")
  cat("      spaces', arXiv:2312.00752; Smith, J.T.H. et al. (2023) 'Simplified\n")
  cat("      state space layers for sequence modeling' (S5), ICLR.\n")
}
