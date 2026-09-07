# Bland-Altman agreement analysis (Reference Sec 13.16)
# Native R via blandr / BlandAltmanLeh; Python via pyCompare / from-scratch.
# Run with:  Rscript bland_altman_agreement.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  blandr                    -- Bland-Altman with LoA CIs, proportional bias\n")
  cat("  BlandAltmanLeh            -- ggplot BA implementation\n")
  cat("  agRee                     -- multi-method + variance-component agreement\n")
  cat("  MethComp                  -- comprehensive method-comparison suite\n")
  cat("Python:\n")
  cat("  pyCompare                -- BA + Bablok Passing analyses\n")
  cat("  scipy + numpy from-scratch (see bland_altman_agreement.py)\n")
  cat("Refs: Bland, J.M. & Altman, D.G. (1986) 'Statistical methods for\n")
  cat("      assessing agreement between two methods of clinical measurement',\n")
  cat("      Lancet 1: 307-310; Bland & Altman (1999) 'Measuring agreement in\n")
  cat("      method comparison studies', Stat Meth Med Res 8(2): 135-160.\n")
}
