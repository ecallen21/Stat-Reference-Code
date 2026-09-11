# Count-Min Sketch (Reference Sec 47.245)
# Native R via bloomfilter / custom hash; Python via datasketch / from-scratch.
# Run with:  Rscript count_min_sketch.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  bloomfilter             -- Bloom + count-min primitives\n")
  cat("  probably (bitmap ops)   -- probabilistic set structures\n")
  cat("  digest::digest          -- hash functions for custom sketches\n")
  cat("Python:\n")
  cat("  datasketch.CountMinSketch\n")
  cat("  probables.countminsketch\n")
  cat("  From-scratch numpy (see count_min_sketch.py)\n")
  cat("Refs: Cormode, G. & Muthukrishnan, S. (2005) 'An Improved Data Stream\n")
  cat("      Summary: The Count-Min Sketch and Its Applications', J Algorithms 55(1).\n")
}
