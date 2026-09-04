# KL divergence (Reference Sec 34.3)
# Native R via FNN; Python via scipy.
# Run with:  Rscript kl_divergence.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  FNN::KL.divergence            -- Kraskov-Stogbauer-Grassberger for continuous\n")
  cat("  entropy::KL.plugin            -- discrete plug-in KL\n")
  cat("  philentropy                   -- 46 distance measures incl. KL / JS / Renyi\n")
  cat("Python:\n")
  cat("  scipy.special.rel_entr         -- element-wise p log(p/q)\n")
  cat("  scipy.stats.entropy(p, q)      -- KL wrapper\n")
  cat("  torch.nn.functional.kl_div     -- differentiable KL for training\n")
  cat("Refs: Kullback, S. & Leibler, R.A. (1951) 'On information and sufficiency',\n")
  cat("      Annals of Math Stat; Lin, J. (1991) 'Divergence measures based on the\n")
  cat("      Shannon entropy' (Jensen-Shannon), IEEE Trans Inf Theory.\n")
}
