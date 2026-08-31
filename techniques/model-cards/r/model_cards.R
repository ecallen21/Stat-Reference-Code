# Model cards (Reference Ch 32 MLOps)
# Native R via yaml + rmarkdown; Python via google model-cards-toolkit.
# Run with:  Rscript model_cards.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  yaml, jsonlite              -- serialise / deserialise model-card metadata\n")
  cat("  rmarkdown / quarto          -- render human-readable model-card PDFs / HTML\n")
  cat("  vetiver                     -- attach model_card metadata to a versioned model\n")
  cat("Python:\n")
  cat("  google model-cards-toolkit  -- reference schema + HTML/PDF export (mct)\n")
  cat("  huggingface transformers    -- ModelCard class + auto-generated card templates\n")
  cat("  mlflow (Description field)   -- lightweight card storage per registered version\n")
  cat("Refs: Mitchell, M. et al. (2019) 'Model Cards for Model Reporting', FAT*.\n")
  cat("      Gebru, T. et al. (2018) 'Datasheets for Datasets', arXiv:1803.09010.\n")
}
