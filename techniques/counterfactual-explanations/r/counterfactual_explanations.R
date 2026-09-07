# Counterfactual explanations (Reference Sec 47.33)
# Native R via iml / lime; Python via alibi / DiCE / CARLA.
# Run with:  Rscript counterfactual_explanations.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  iml::Counterfactuals             -- MOC multi-objective counterfactual search\n")
  cat("  lime + custom modification loops\n")
  cat("Python:\n")
  cat("  alibi.explainers.CounterfactualProto / Counterfactual (Seldon)\n")
  cat("  DiCE (Diverse Counterfactual Explanations, Microsoft)\n")
  cat("  CARLA (benchmarking counterfactual algorithms)\n")
  cat("Refs: Wachter, S., Mittelstadt, B. & Russell, C. (2017) 'Counterfactual\n")
  cat("      explanations without opening the black box: automated decisions\n")
  cat("      and the GDPR', Harvard J Law & Tech 31; Mothilal, R.K., Sharma, A.\n")
  cat("      & Tan, C. (2020) 'Explaining machine learning classifiers through\n")
  cat("      diverse counterfactual explanations' (DiCE), FAT.\n")
}
