# Online learning via SGD (Reference §21.x extra)
# R has less-developed streaming ML support; use bigmemory + biglm or Python sklearn.
# Run with:  Rscript online_learning_sgd.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  biglm::biglm / bigglm                -- one-pass linear / GLM on chunked data\n")
  cat("  RSGD::rsgd                            -- Toulis-Airoldi implicit SGD\n")
  cat("  RcppRoll::roll_ family                -- rolling / expanding-window updates\n")
  cat("  stream::DSD / DSC                     -- streaming clustering & classifiers\n")
  cat("Python:\n")
  cat("  sklearn.linear_model.SGDRegressor / SGDClassifier / PassiveAggressive*\n")
  cat("  river (formerly creme)                -- production-grade online ML library\n")
}
