# Reptile meta-learning (Reference Sec 47.134)
# Python via learn2learn / higher / torchmeta.
# Run with:  Rscript reptile_meta_learning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class meta-learning suite in R\n")
  cat("  torch (R)                     -- roll your own Reptile / MAML loops\n")
  cat("Python:\n")
  cat("  learn2learn.algorithms.Reptile / MAML   -- reference PyTorch library\n")
  cat("  higher                                 -- Facebook AI: differentiable optimisers\n")
  cat("  torchmeta                              -- meta-learning datasets + wrappers\n")
  cat("  jax.experimental.optimizers            -- Reptile in JAX\n")
  cat("  from-scratch                           -- see reptile_meta_learning.py\n")
  cat("Refs: Nichol, Achiam & Schulman (2018) 'On first-order meta-learning\n")
  cat("      algorithms', arXiv:1803.02999; Finn, Abbeel & Levine (2017) 'MAML',\n")
  cat("      ICML.\n")
}
