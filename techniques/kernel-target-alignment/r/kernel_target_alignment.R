# Kernel Target Alignment (Reference Sec 47.136)
# Native R via kernlab; Python via sklearn / custom.
# Run with:  Rscript kernel_target_alignment.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  kernlab::kernelPol / rbfdot   -- kernel matrices; alignment via custom\n")
  cat("  mklaren                       -- multi-kernel learning with alignment weights\n")
  cat("Python:\n")
  cat("  sklearn.metrics.pairwise      -- kernel matrices\n")
  cat("  mklearn / EasyMKL             -- MKL libraries using KTA\n")
  cat("  from-scratch                  -- see kernel_target_alignment.py\n")
  cat("Refs: Cristianini, Kandola, Elisseeff & Shawe-Taylor (2001) NeurIPS;\n")
  cat("      Cortes, Mohri & Rostamizadeh (2012) 'Algorithms for learning kernels\n")
  cat("      based on centered alignment', JMLR 13.\n")
}
