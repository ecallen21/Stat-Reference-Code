# f-Divergences (Reference Sec 34.7)
# Native R via philentropy; Python via scipy / POT.
# Run with:  Rscript f_divergences.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  philentropy                  -- 46 distance measures including all f-divergences\n")
  cat("  transport                     -- Wasserstein / optimal transport\n")
  cat("  FNN                            -- k-NN KL and Renyi\n")
  cat("Python:\n")
  cat("  scipy.special.rel_entr / scipy.stats.entropy\n")
  cat("  POT (Python Optimal Transport) -- Wasserstein, Sinkhorn\n")
  cat("  torch.distributions.kl.kl_divergence for standard families\n")
  cat("Refs: Csiszar, I. (1967) 'Information-type measures of difference of\n")
  cat("      probability distributions and indirect observations', Studia Scientiarum\n")
  cat("      Mathematicarum Hungarica; Liese, F. & Vajda, I. (2006) 'On divergences\n")
  cat("      and informations in statistics and information theory', IEEE Trans IT.\n")
}
