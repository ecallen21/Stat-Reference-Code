# Lion — EvoLved Sign Momentum (Chen et al 2023)
# R: `torch::optim_ignite_lion` (via torch bindings) — no
#    first-class R package; usually access via reticulate to
#    `pypi lion-pytorch` or `keras-cv-attention-models`.
# Python: `lion-pytorch::Lion`, `keras.optimizers.Lion`,
#         `optax.contrib.lion`, from-scratch
#
# reticulate::py_install("lion-pytorch")
# lion <- reticulate::import("lion_pytorch")$Lion
# opt <- lion(model$parameters, lr = 1e-4, weight_decay = 1e-2)
