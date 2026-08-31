# Question answering (Reference §25.x extra)
# R via reticulate + Python (huggingface transformers, haystack).
# Run with:  Rscript question_answering.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  huggingface transformers.pipeline('question-answering', model='deepset/roberta-base-squad2')\n")
  cat("  haystack (deepset) -- production QA + retrieval-augmented QA\n")
  cat("  langchain / llama-index -- RAG pipelines with LLM readers\n")
  cat("Datasets: SQuAD 1.1/2.0, TriviaQA, Natural Questions, HotpotQA, MS MARCO.\n")
  cat("Family:\n")
  cat("  * Extractive QA (span selection)      -- BERT + start/end heads\n")
  cat("  * Abstractive QA                       -- seq2seq generator (T5, BART)\n")
  cat("  * Retrieval-augmented (RAG)            -- retriever + generator LLM\n")
  cat("  * ReAct / Toolformer                   -- LLM + tool use (search, calculator)\n")
  cat("  * Multi-hop reasoning                  -- decompose Q into subqueries (Chain-of-Thought, HotpotQA)\n")
}
