# Retrieval-augmented generation (RAG) (Reference Sec 47.19)
# Native R via ellmer / rag; Python via LangChain / LlamaIndex.
# Run with:  Rscript retrieval_augmented_generation.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  ellmer                   -- Posit LLM client, supports RAG assemblies\n")
  cat("  chattr                    -- shell chat wrappers, retrieval helpers\n")
  cat("  quanteda / text2vec      -- retrieval building blocks (TF-IDF, embeddings)\n")
  cat("Python:\n")
  cat("  langchain / langchain-community    -- retrievers + LLM chains\n")
  cat("  llama-index                          -- indexing + query engines\n")
  cat("  chromadb / weaviate / qdrant         -- vector databases\n")
  cat("  sentence-transformers                -- dense embeddings for retrieval\n")
  cat("  transformers                         -- generator LLM\n")
  cat("Refs: Lewis, P. et al. (2020) 'Retrieval-augmented generation for\n")
  cat("      knowledge-intensive NLP tasks', NeurIPS 33; Karpukhin, V. et al.\n")
  cat("      (2020) 'Dense passage retrieval for open-domain question answering',\n")
  cat("      EMNLP.\n")
}
