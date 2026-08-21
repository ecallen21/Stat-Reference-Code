# Entity embeddings for categorical features (Reference §27.12)
# R via torch or keras3.
# Run with:  Rscript embedding_layers.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  torch::nn_embedding(num_embeddings=K, embedding_dim=d)\n")
  cat("  keras3::layer_embedding(input_dim=K, output_dim=d)\n")
  cat("  tabnet / xrf / fastai-style entity-embedding pipelines wrap this for tabular tasks.\n")
  cat("Recipes: standardise or one-hot small K categories; embed large-K categoricals\n")
  cat("         (zip codes, product SKUs, user IDs).  dim ~ min(50, ceil(K^0.25)) rule of thumb.\n")
  cat("Python: torch.nn.Embedding, tensorflow.keras.layers.Embedding, fastai TabularModel.\n")
}
