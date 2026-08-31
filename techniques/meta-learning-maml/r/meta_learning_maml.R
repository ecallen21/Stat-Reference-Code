# Model-Agnostic Meta-Learning — MAML (Reference §27.x extra)
# R via reticulate + Python.
# Run with:  Rscript meta_learning_maml.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  learn2learn  -- PyTorch meta-learning library (MAML, ANIL, ProtoNets, etc.)\n")
  cat("  higher        -- PyTorch second-order autograd wrapper for MAML\n")
  cat("  torchmeta     -- benchmarks + loaders for few-shot problems\n")
  cat("  JAX: jax-metalearning\n")
  cat("Family:\n")
  cat("  * MAML (Finn 2017)                    -- second-order (full) meta gradient\n")
  cat("  * First-order MAML (FOMAML)           -- drops the second-order term\n")
  cat("  * Reptile (Nichol 2018)                -- simple SGD analogue of MAML\n")
  cat("  * ANIL (Raghu 2019)                    -- adapt only the head\n")
  cat("  * ProtoNets (Snell 2017)               -- metric-based few-shot classification\n")
  cat("  * Matching Networks (Vinyals 2016)     -- attention-based few-shot\n")
  cat("  * Meta-Baseline (Chen 2020)           -- pretrain + fine-tune is surprisingly strong\n")
}
