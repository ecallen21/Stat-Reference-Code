# Hoeffding's D independence test (Reference Sec 47.133)
# Native R via Hmisc; Python via hyppo / from-scratch.
# Run with:  Rscript hoeffding_d_independence.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Hmisc::hoeffd            -- reference implementation with permutation p\n")
  cat("  energy::dcor.test        -- distance correlation alternative\n")
  cat("  minerva::mine            -- MIC / MAS / MEV nonparametric\n")
  cat("Python:\n")
  cat("  hyppo.independence.Hsic  -- kernel independence (HSIC)\n")
  cat("  scipy.stats.spearmanr / kendalltau -- monotonic only\n")
  cat("  from-scratch             -- see hoeffding_d_independence.py\n")
  cat("Refs: Hoeffding (1948) Ann Math Stat 19(4); Hollander & Wolfe (1999)\n")
  cat("      'Nonparametric Statistical Methods', 2nd ed, Wiley.\n")
}
