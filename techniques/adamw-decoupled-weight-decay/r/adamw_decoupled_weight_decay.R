# AdamW — Decoupled Weight Decay (Loshchilov-Hutter 2019)
# R: `torch::optim_adamw`, `keras::optimizer_adamw`
# Python: `torch.optim.AdamW`, `keras.optimizers.AdamW`,
#         `optax.adamw`, from-scratch
#
# library(torch)
# opt <- optim_adamw(model$parameters, lr = 1e-3, weight_decay = 0.01)
