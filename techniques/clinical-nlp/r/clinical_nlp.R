# Clinical NLP (Reference Sec 42.5)
# Native pipelines cTAKES / MedSpaCy; R clinspacy; Python scispacy + negspacy.
# Run with:  Rscript clinical_nlp.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  clinspacy                       -- MedSpaCy via reticulate + UMLS linking\n")
  cat("  spacyr                          -- generic spaCy interface\n")
  cat("  tidytext                        -- generic tokenisation\n")
  cat("Python:\n")
  cat("  scispacy + models (en_core_sci_sm, en_ner_bc5cdr_md)\n")
  cat("  medspacy (Chapman NegEx + ConText + sectionizer)\n")
  cat("  negspacy                        -- pipeline component for negation\n")
  cat("  transformers::BioBERT / ClinicalBERT\n")
  cat("External:\n")
  cat("  Apache cTAKES                    -- clinical concept extractor with UMLS\n")
  cat("Refs: Savova et al. (2010) 'cTAKES', JAMIA; Chapman et al. (2001) 'A simple\n")
  cat("      algorithm for identifying negated findings and diseases in discharge\n")
  cat("      summaries', J Biomed Inform (NegEx); Alsentzer et al. (2019) 'Publicly\n")
  cat("      available clinical BERT embeddings', Clinical NLP Workshop.\n")
}
