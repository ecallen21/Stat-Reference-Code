# Gumbel-Softmax / Concrete (Reference Sec 47.315)
# Native R via torch (R) + custom; Python via torch.nn.functional / TFP / from-scratch.
# Run with:  Rscript gumbel_softmax_relaxation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch (R) with -log(-log(U)) + softmax((logits+g)/tau)\n")
  cat("  reticulate + torch\n")
  cat("Python:\n")
  cat("  torch.nn.functional.gumbel_softmax(logits, tau, hard=False)\n")
  cat("  tensorflow_probability.distributions.RelaxedOneHotCategorical\n")
  cat("  Pyro / NumPyro (categorical RelaxedOneHot)\n")
  cat("  From-scratch (see gumbel_softmax_relaxation.py)\n")
  cat("Refs: Jang, E., Gu, S. & Poole, B. (2017) 'Categorical reparameterization\n")
  cat("      with Gumbel-Softmax', ICLR;  Maddison, C.J., Mnih, A. & Teh, Y.W.\n")
  cat("      (2017) 'The Concrete distribution', ICLR.\n")
}
