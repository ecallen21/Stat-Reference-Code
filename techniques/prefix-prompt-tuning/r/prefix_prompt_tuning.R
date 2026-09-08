# Prefix / Prompt Tuning (Reference Sec 47.111)
# Python-only; peft (HF) is the reference.
# Run with:  Rscript prefix_prompt_tuning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  no first-class prompt-tuning in R; call Python via reticulate\n")
  cat("Python:\n")
  cat("  peft.PromptTuningConfig / PrefixTuningConfig -- HF reference\n")
  cat("  peft.LoraConfig                              -- LoRA adapter alternative\n")
  cat("  peft.PromptEncoderConfig                     -- soft-prompt encoder\n")
  cat("  transformers.Trainer + PEFT                  -- fine-tuning pipeline\n")
  cat("  from-scratch analogue                        -- see prefix_prompt_tuning.py\n")
  cat("Refs: Li & Liang (2021) 'Prefix-tuning', ACL; Lester, Al-Rfou & Constant\n")
  cat("      (2021) 'The power of scale for parameter-efficient prompt tuning',\n")
  cat("      EMNLP.\n")
}
