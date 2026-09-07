# Deep survival networks (Reference Sec 11.27)
# Native R via survivalmodels; Python via pycox / auton-survival.
# Run with:  Rscript deep_survival_network.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survivalmodels             -- DeepSurv, DeepHit, PC-Hazard, Coxtime wrappers\n")
  cat("  R6 + torch                  -- from-scratch DeepSurv via torch-for-R\n")
  cat("  mlr3proba                    -- probabilistic survival machine-learning fw\n")
  cat("Python:\n")
  cat("  pycox                      -- Katzman DeepSurv + DeepHit + Coxtime + PC-Haz\n")
  cat("  auton-survival             -- CMU library incl. DeepSurv / DSM / DCM\n")
  cat("  sksurv.linear_model / ensemble  -- shallow baselines for comparison\n")
  cat("  torchsurv                    -- PyTorch survival losses / metrics\n")
  cat("Refs: Katzman, J.L. et al. (2018) 'DeepSurv: personalized treatment\n")
  cat("      recommender system using a Cox proportional hazards deep neural\n")
  cat("      network', BMC Med Res Meth 18: 24; Lee, C. et al. (2018) 'DeepHit:\n")
  cat("      a deep learning approach to survival analysis with competing risks',\n")
  cat("      AAAI.\n")
}
