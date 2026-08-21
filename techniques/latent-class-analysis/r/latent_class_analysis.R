# Latent Class Analysis (Reference §19.x extra)
# R via poLCA, depmixS4, or lavaan (limited).
# Run with:  Rscript latent_class_analysis.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  poLCA::poLCA(f, data, nclass=K, nrep=10)      -- classical LCA with restarts\n")
  cat("  poLCA::poLCA.bootstrap(...)                    -- bootstrap SEs and posteriors\n")
  cat("  depmixS4::mix / fit                            -- LCA + LCM including continuous items\n")
  cat("  Mclust from mclust (for continuous latent-profile analysis, LPA)\n")
  cat("  lavaan (limited categorical latent variables via WLSMV)\n")
  cat("  scikit-learn / stepmix (Python) — supervised LCA / hidden-Markov extensions\n")
}
