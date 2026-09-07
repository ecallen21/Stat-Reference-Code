# U-statistics (Reference Sec 46.9)
# Native R via Ustat; Python via from-scratch numpy.
# Run with:  Rscript u_statistics.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Ustat / uStats            -- generic U-statistics with variance formulas\n")
  cat("  ineq::Gini                -- Gini index (2nd-order U kernel)\n")
  cat("  ape::mantel.test          -- Mantel U for distance matrices\n")
  cat("  Kendall::Kendall          -- Kendall's tau (2nd-order U)\n")
  cat("Python:\n")
  cat("  scipy.stats.kendalltau     -- Kendall tau as a normalised U-statistic\n")
  cat("  from-scratch: itertools.combinations + numpy for exact / vectorised\n")
  cat("Refs: Hoeffding, W. (1948) 'A class of statistics with asymptotically normal\n")
  cat("      distribution', Ann Math Stat 19(3): 293-325; Serfling (1980)\n")
  cat("      Approximation Theorems of Mathematical Statistics, Wiley.\n")
}
