# EM algorithm for finite mixtures (Reference Sec 47.77)
# Native R via mclust / mixtools; Python via sklearn.
# Run with:  Rscript em_algorithm_mixture.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mclust::Mclust                  -- GMM + model selection over 14 shapes\n")
  cat("  mixtools::normalmixEM / gammamixEM -- univariate mixtures\n")
  cat("  flexmix                         -- mixture regression / mixed-effects\n")
  cat("  EMMIXcskew                      -- skew-t mixture EM\n")
  cat("Python:\n")
  cat("  sklearn.mixture.GaussianMixture / BayesianGaussianMixture\n")
  cat("  pomegranate.GeneralMixtureModel -- any distribution family\n")
  cat("  from-scratch                    -- see em_algorithm_mixture.py\n")
  cat("Refs: Dempster, Laird & Rubin (1977) JRSS-B 39(1); McLachlan & Krishnan\n")
  cat("      (2008) 'The EM Algorithm and Extensions', 2nd ed, Wiley.\n")
}
