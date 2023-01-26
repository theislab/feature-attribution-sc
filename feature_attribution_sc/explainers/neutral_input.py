from copy import deepcopy

import torch


def neutral_input(scanvi_model, epochs=2000, batch_index=None, size_factor=None, **kwargs):
    scanvi_model = deepcopy(scanvi_model)
    for param in scanvi_model.module.parameters():
        param.requires_grad = False

    if batch_index is not None and not isinstance(batch_index, torch.Tensor):
        batch_index = torch.tensor(batch_index, dtype=torch.float32, device=param.device).reshape((1, 1))

    n_vars = scanvi_model.summary_stats["n_vars"]
    inpt = torch.ones((1, n_vars), device=param.device)
    if size_factor is None:
        inpt = inpt * 0.0
    else:
        inpt = inpt / n_vars
        inpt = torch.distributions.Dirichlet(inpt).sample() * size_factor
    inpt.requires_grad = True

    optim = torch.optim.Adam(params=(inpt,), **kwargs)

    for i in range(epochs):
        optim.zero_grad()
        probs = scanvi_model.module.classify(x=inpt, batch_index=batch_index)
        if i % 100 == 0:
            print(f"Epoch {i}: max probability - {probs.max().item()}.")
        entropy = torch.distributions.Categorical(probs=probs).entropy()
        (-entropy).backward()
        optim.step()
        inpt.data.clamp_(0.0)

    return inpt.detach()
