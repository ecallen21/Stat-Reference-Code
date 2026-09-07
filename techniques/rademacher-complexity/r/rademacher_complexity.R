# Rademacher complexity (Reference Sec 46.10)
# From-scratch in both R and Python (no first-class library).
# Run with:  Rscript rademacher_complexity.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  No dedicated CRAN package -- implement from scratch\n")
  cat("  Related: RSSampling, mlr3verse (learning-curve utilities)\n")
  cat("Python:\n")
  cat("  From-scratch numpy (see accompanying rademacher_complexity.py)\n")
  cat("  Related: scikit-learn learning-curve, mlxtend for VC/Rademacher estimation\n")
  cat("Refs: Bartlett, P.L. & Mendelson, S. (2002) 'Rademacher and Gaussian\n")
  cat("      complexities: risk bounds and structural results', JMLR 3: 463-482;\n")
  cat("      Mohri, Rostamizadeh & Talwalkar (2018) Foundations of Machine\n")
  cat("      Learning, 2nd ed., MIT Press.\n")
}
