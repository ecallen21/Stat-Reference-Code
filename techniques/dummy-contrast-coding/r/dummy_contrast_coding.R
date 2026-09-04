# Dummy / effect / contrast coding (Reference Sec 41.6)
# Native R via stats::contr.*; Python patsy + custom.
# Run with:  Rscript dummy_contrast_coding.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::contr.treatment / contr.sum / contr.helmert / contr.poly / contr.SAS\n")
  cat("  stats::model.matrix              -- expand a factor with chosen contrast\n")
  cat("  fastDummies::dummy_cols          -- dummy expansion\n")
  cat("  recipes::step_dummy               -- pipeline step\n")
  cat("Python:\n")
  cat("  patsy::C(x, Treatment / Sum / Helmert / Poly / Diff)\n")
  cat("  pandas.get_dummies                -- dummy coding\n")
  cat("  sklearn.preprocessing.OneHotEncoder\n")
  cat("Refs: Cohen, Cohen, West & Aiken (2003) Applied Multiple Regression/Correlation\n")
  cat("      Analysis, 3rd ed., Routledge Ch 8; Davis (2010) 'Contrast coding in\n")
  cat("      multiple regression analysis', J Data Sci.\n")
}
