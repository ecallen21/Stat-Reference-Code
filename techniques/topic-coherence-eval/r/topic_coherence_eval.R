# Topic-coherence metrics (Reference §25.11)
# R via textmineR, quanteda, or topicdoc.
# Run with:  Rscript topic_coherence_eval.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  textmineR::CalcProbCoherence(phi, dtm, M=6)             -- UMass-style coherence per topic\n")
  cat("  topicdoc::topic_coherence(model, dtm, top_n=6)          -- multiple coherence metrics\n")
  cat("  quanteda + textmodels workflows for LDA + evaluation\n")
  cat("Python:\n")
  cat("  gensim.models.CoherenceModel(topics, texts, dictionary, coherence='u_mass' | 'c_v' | 'c_uci' | 'c_npmi')\n")
  cat("Human evaluation (word intrusion, topic intrusion) — Chang et al. 2009 — remains the gold standard.\n")
}
