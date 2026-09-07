# Wilson / Agresti-Coull / Jeffreys CIs for proportions (Reference Sec 4.18)
# Native R via binom / PropCIs; Python via statsmodels.stats.proportion.
# Run with:  Rscript wilson_score_interval_proportion.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  binom::binom.confint     -- Wilson / Agresti-Coull / Jeffreys / CP / Wald\n")
  cat("  PropCIs                   -- extended CI catalogue incl. mid-p, Blaker\n")
  cat("  stats::prop.test          -- built-in Wilson (correct = TRUE default)\n")
  cat("Python:\n")
  cat("  statsmodels.stats.proportion.proportion_confint(method='wilson' etc.)\n")
  cat("  scipy.stats.binomtest with proportion_ci method\n")
  cat("  from-scratch (see wilson_score_interval_proportion.py)\n")
  cat("Refs: Wilson, E.B. (1927) 'Probable inference, the law of succession, and\n")
  cat("      statistical inference', JASA 22(158): 209-212; Agresti, A. & Coull,\n")
  cat("      B.A. (1998) 'Approximate is better than exact for interval estimation\n")
  cat("      of binomial proportions', Am Statist 52(2); Brown, Cai & DasGupta\n")
  cat("      (2001) 'Interval estimation for a binomial proportion', Stat Sci.\n")
}
