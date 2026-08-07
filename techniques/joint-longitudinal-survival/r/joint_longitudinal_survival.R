# Joint longitudinal-survival model (Reference §12.10)
# R via JM::jointModel or JMbayes2::jm.
# Run with:  Rscript joint_longitudinal_survival.R

if (sys.nframe() == 0) {
  cat("The canonical R packages for joint models:\n")
  cat("  - JM::jointModel  (MLE)\n")
  cat("  - JMbayes2::jm    (Bayesian; supports multiple biomarkers)\n\n")
  cat("Typical call structure:\n")
  cat("  lme_fit <- nlme::lme(y ~ time, random = ~ time | id, data = long)\n")
  cat("  cox_fit <- survival::coxph(Surv(time, event) ~ 1, data = surv, x = TRUE)\n")
  cat("  jm_fit  <- JM::jointModel(lme_fit, cox_fit, timeVar = \"time\",\n")
  cat("                            method = \"weibull-PH-aGH\")\n")
}
