# Conditional Random Field (Reference Sec 47.157).
#
# Lafferty, McCallum & Pereira 2001. Discriminative sequence
# labeler; log-linear over the full sequence.
#
# Uses `crfsuite` (R wrapper for CRFsuite).

if (!requireNamespace("crfsuite", quietly = TRUE)) install.packages("crfsuite")
library(crfsuite)                                     # CRFsuite bindings

cat("=== CRF (Lafferty-McCallum-Pereira 2001) ===\n")
cat("Use `crfsuite::crf` to fit linear-chain CRFs on token-level features.\n")
cat("A typical workflow:\n")
cat("  crf_model <- crf(y = tags, x = features, group = sentence_id,\n")
cat("                    method = 'lbfgs', options = list(max_iterations = 50))\n")
cat("Then predict(crf_model, newdata = new_features, group = new_sentence_id).\n")
