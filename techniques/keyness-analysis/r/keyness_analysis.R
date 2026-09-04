# Keyness analysis (Reference Sec 42.15)
# Native R via quanteda::textstat_keyness; Python textacy + custom.
# Run with:  Rscript keyness_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quanteda::textstat_keyness      -- G^2 / chi^2 / BIC / LR keyness\n")
  cat("  quanteda::tokens_keep + textstat_frequency\n")
  cat("Python:\n")
  cat("  textacy.keyterms                 -- corpus-level keyness / keyterm extraction\n")
  cat("  sklearn + scipy.stats.chi2       -- custom G^2 / chi^2\n")
  cat("Refs: Dunning (1993) 'Accurate methods for the statistics of surprise and\n")
  cat("      coincidence', Comput Ling; Gabrielatos & Marchi (2012) 'Keyness:\n")
  cat("      appropriate metrics and practical issues', CADS.\n")
}
