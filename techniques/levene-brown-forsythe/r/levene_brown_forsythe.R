# Levene / Brown-Forsythe variance test (Reference Sec 47.61)
# Native R via car / onewaytests; Python via scipy.stats.
# Run with:  Rscript levene_brown_forsythe.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  car::leveneTest(y ~ group, center=median)  -- Brown-Forsythe median\n")
  cat("  onewaytests::bf.test                        -- BF-mean/median/trimmed\n")
  cat("  stats::bartlett.test                        -- Bartlett baseline\n")
  cat("  stats::fligner.test                         -- rank-based alternative\n")
  cat("Python:\n")
  cat("  scipy.stats.levene(*groups, center='median') -- built-in BF-median\n")
  cat("  scipy.stats.fligner                          -- Fligner-Killeen\n")
  cat("  from-scratch                                 -- see levene_brown_forsythe.py\n")
  cat("Refs: Levene (1960) in 'Contributions to Probability and Statistics';\n")
  cat("      Brown & Forsythe (1974) JASA 69(346).\n")
}
