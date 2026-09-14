# Shampoo (Gupta-Koren-Singer 2018)
# R: no first-class R package; access via reticulate to
#    `google-research/scalable_shampoo` or `optax.distributed_shampoo`.
# Python: `optax.scale_by_shampoo`, `pytorch-optimizer.Shampoo`,
#         Google's `scalable_shampoo` (JAX+TF), from-scratch
#
# reticulate::py_install("scalable-shampoo")
# shampoo <- reticulate::import("scalable_shampoo")
