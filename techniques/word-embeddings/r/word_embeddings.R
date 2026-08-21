# Word embeddings (Reference §25.3)
# R via text2vec (GloVe), wordVectors, or fastTextR.
# Run with:  Rscript word_embeddings.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  text2vec::GloVe$new(rank=50, x_max=10)  +  fit_transform(tcm, ...)  -- GloVe\n")
  cat("  wordVectors::train_word2vec('corpus.txt', vectors=100, threads=4)   -- CBOW / SGNS\n")
  cat("  fastTextR::ft_train(...)                                             -- FastText (subword)\n")
  cat("  textdata::embedding_glove6b(...)                                    -- pretrained GloVe download\n")
  cat("Python: gensim.models.Word2Vec (CBOW/SGNS), fasttext, spaCy nlp.vocab.vectors,\n")
  cat("        sentence-transformers for contextual dense embeddings (BERT etc).\n")
}
