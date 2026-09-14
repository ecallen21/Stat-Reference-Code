# LAMB (You et al 2020)
# R: no first-class R package; access via reticulate to
#    `torch_optimizer.Lamb` or `nvidia.LAMB`.
# Python: `torch_optimizer.Lamb`, `nvidia.LAMB`,
#         `keras.optimizers.LAMB`, `deepspeed.ops.lamb.FusedLamb`,
#         from-scratch
#
# reticulate::py_install("torch_optimizer")
# lamb <- reticulate::import("torch_optimizer")$Lamb
