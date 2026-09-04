# Structural topic model (Reference Sec 42.7)
# Native R via stm; Python custom + bertopic + rpy2.
# Run with:  Rscript structural_topic_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stm::stm + estimateEffect + findTopics -- Roberts-Stewart-Tingley STM\n")
  cat("  stmBrowser                       -- interactive STM visualisation\n")
  cat("Python:\n")
  cat("  stm (via rpy2)                   -- STM reference implementation\n")
  cat("  bertopic                         -- BERT-based topic model with metadata\n")
  cat("  gensim                            -- baseline LDA\n")
  cat("Refs: Roberts, Stewart & Tingley (2019) 'stm: An R package for structural topic\n")
  cat("      models', JSS 91(2); Roberts et al. (2014) 'Structural topic models for\n")
  cat("      open-ended survey responses', AJPS.\n")
}
