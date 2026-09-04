# Discretization / binning (Reference Sec 41.7)
# Native R via arules / Hmisc; Python sklearn + custom.
# Run with:  Rscript discretization_binning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  arules::discretize              -- equal-freq / equal-width / fixed / cluster\n")
  cat("  Hmisc::cut2                     -- quantile-based cut points\n")
  cat("  infotheo::discretize            -- entropy-based binning\n")
  cat("Python:\n")
  cat("  sklearn.preprocessing.KBinsDiscretizer (uniform / quantile / kmeans)\n")
  cat("  pandas.cut / qcut               -- cutpoint / quantile-based\n")
  cat("  optbinning                      -- MDL / IV-based binning (Fayyad-Irani)\n")
  cat("Refs: Royston, Altman & Sauerbrei (2006) 'Dichotomizing continuous predictors\n")
  cat("      in multiple regression: a bad idea', Stat Med; Fayyad & Irani (1993)\n")
  cat("      'Multi-interval discretization of continuous-valued attributes', IJCAI.\n")
}
