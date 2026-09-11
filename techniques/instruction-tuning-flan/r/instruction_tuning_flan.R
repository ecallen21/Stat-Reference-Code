# Instruction Tuning / FLAN (Reference Sec 47.182).
#
# Wei et al 2022. Fine-tune a pretrained LM on many tasks reformulated
# as natural-language instructions; unlocks zero-shot generalisation.

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)                                   # bridge to Python

cat("=== Instruction Tuning / FLAN (Wei et al 2022) ===\n")
cat("No native R impl. Recommended libraries (Python):\n")
cat("  * transformers.Trainer / SFTTrainer (trl)\n")
cat("  * FLAN-T5 / T0 / Tulu / OpenOrca datasets on HF Hub\n")
cat("  * axolotl, LLaMA-Factory (fine-tuning frameworks)\n")
