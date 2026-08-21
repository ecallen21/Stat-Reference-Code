# Document clustering (Reference §25.5)
# R via text2vec, quanteda, cluster, or skmeans.
# Run with:  Rscript document_clustering.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  text2vec::create_dtm + TfIdf$new  +  skmeans::skmeans(dtm, k)   -- spherical k-means\n")
  cat("  quanteda::dfm_tfidf + textmineR::CalcTopicModelR2                -- doc clustering suite\n")
  cat("  cluster::pam(dist_matrix, k)                                     -- k-medoids on cosine dist\n")
  cat("  mclust::Mclust                                                    -- Gaussian-mixture clustering on dense embeddings\n")
  cat("  aricode::NMI / ARI                                                -- evaluation metrics\n")
  cat("Python: sklearn.cluster.KMeans / MiniBatchKMeans, sklearn.metrics for NMI/ARI.\n")
}
