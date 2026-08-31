# Relation extraction (Reference §25.x extra)
# R via reticulate + Python.
# Run with:  Rscript relation_extraction.R

if (sys.nframe() == 0) {
  cat("R packages: no strong native R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  OpenNRE  (Thunlp)  -- neural relation extraction with pretrained models\n")
  cat("  spaCy REL component -- built-in relation extractor for spaCy pipelines\n")
  cat("  DeepKE (Zhejiang) -- knowledge extraction toolkit; RE, NER, event extraction\n")
  cat("  huggingface transformers -- fine-tune BERT / DeBERTa for TACRED / SemEval-2010 Task 8\n")
  cat("Approaches:\n")
  cat("  * Rule / pattern-based (Hearst 1992)   -- simple regex; interpretable\n")
  cat("  * Distant supervision (Mintz 2009)     -- align KB tuples to text\n")
  cat("  * Neural: PCNN (Zeng 2015), BERT-EM (Soares 2019), REBEL (Cabot 2021)\n")
  cat("  * Joint NER + RE: JEREX, PL-Marker, W2NER\n")
  cat("  * Open information extraction: OpenIE, DrKIT.\n")
}
