# Whisper (Reference Sec 47.231).
#
# Radford et al 2022 OpenAI. Encoder-decoder Transformer for ASR
# + translation trained on 680k hrs of weakly-supervised web audio.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Whisper (Radford et al 2022) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * openai-whisper (reference PyTorch impl)\n")
cat("  * faster-whisper (CTranslate2 for CPU / GPU speedup)\n")
cat("  * distil-whisper (distilled, smaller)\n")
cat("  * transformers.WhisperForConditionalGeneration\n")
