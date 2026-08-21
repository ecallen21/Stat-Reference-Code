# Latent Dirichlet Allocation (Reference §25.4)
# R via topicmodels, lda, or stm.
# Run with:  Rscript topic_modeling_lda.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  topicmodels::LDA(dtm, k = K, method = 'Gibbs')       -- collapsed Gibbs\n")
  cat("  topicmodels::LDA(dtm, k = K, method = 'VEM')         -- variational EM (Blei et al.)\n")
  cat("  lda::lda.collapsed.gibbs.sampler(...)                 -- older Gibbs API\n")
  cat("  stm::stm(documents, vocab, K, prevalence = ~ meta)   -- structural topic model with covariates\n")
  cat("  quanteda + textmodels::textmodel_lda                  -- quanteda-native LDA\n")
  cat("Python: sklearn.decomposition.LatentDirichletAllocation, gensim.models.LdaModel,\n")
  cat("        BERTopic (transformer embeddings + clustering).\n")
}
