# Friedman's H-statistic (Reference Sec 47.95)
# Native R via iml / pre; Python via sklearn.inspection + custom.
# Run with:  Rscript friedmans_h_statistic.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  iml::Interaction         -- Molnar's H^2 for pairs and singles\n")
  cat("  pre                      -- Rule-ensemble learner + H-statistic\n")
  cat("  DALEX::model_parts       -- model-agnostic explanations\n")
  cat("Python:\n")
  cat("  sklearn.inspection.partial_dependence  -- PD building block\n")
  cat("  interpret / dalex                       -- H-stat wrappers\n")
  cat("  from-scratch                            -- see friedmans_h_statistic.py\n")
  cat("Refs: Friedman & Popescu (2008) 'Predictive learning via rule ensembles',\n")
  cat("      Ann Appl Stat 2(3).\n")
}
