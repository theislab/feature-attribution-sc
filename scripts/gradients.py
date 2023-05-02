import torch


def batch_to_dict_scanvi(batch):
    return dict(x=batch["X"], batch_index=batch["batch"])


def batch_jacobian(outpt, inpt):
    n_out = outpt.shape[-1]

    jacs = []

    ones = torch.ones(outpt.shape[0]).to(outpt.device)

    for i in range(n_out):
        retain_graph = i != n_out - 1
        jacs.append(torch.autograd.grad(outpt[..., i], inpt, ones, retain_graph=retain_graph)[0])

    return torch.stack(jacs, dim=-1)


def expected_jacobian_step(func, inpt_dict, backprop_inpt_key, prime_inpt):
    x = inpt_dict[backprop_inpt_key]
    x_prime = prime_inpt.to(x.device)

    unif_coef = torch.rand(x.shape[0])[:, None].to(x.device)

    x_diff = x - x_prime

    new_inpt_dict = dict(inpt_dict)
    new_inpt_dict[backprop_inpt_key] = x_prime + x_diff * unif_coef

    out = func(**new_inpt_dict)

    jac = batch_jacobian(out, x)

    return jac * x_diff[..., None].detach()


def integrated_jacobian(func, inpt_dict, backprop_inpt_key, prime_inpt=None, n_steps=10):
    x = inpt_dict[backprop_inpt_key]
    if prime_inpt is not None:
        x_prime = prime_inpt.to(x.device)
    else:
        x_prime = torch.zeros_like(x)

    x_diff = x - x_prime

    jacs = []

    new_inpt_dict = dict(inpt_dict)

    for i in range(n_steps):
        new_inpt_dict[backprop_inpt_key] = x_prime + x_diff * (i + 1) / n_steps

        out = func(new_inpt_dict['x'])

        jacs.append(batch_jacobian(out, x))

    return sum(jacs) * (1 / n_steps) * x_diff[..., None].detach()


def run_expected_jacobian_scanvi(module_func, dl_base, dl_prime, n_steps=10, apply_abs=False, sum_obs=False):
    expected_jacs = []

    for batch in dl_base:
        inpt_dict = batch_to_dict_scanvi(batch)
        inpt_dict["x"].requires_grad = True
#        inpt_dict["x"] = inpt_dict["x"].cuda()

        inpt_batch_size = inpt_dict["x"].shape[0]

        jacs_batch = []
        for i, batch_prime in enumerate(dl_prime):
            if i > n_steps - 1:
                break

            prime_inpt = batch_to_dict_scanvi(batch_prime)["x"]
            if inpt_batch_size < prime_inpt.shape[0]:
                prime_inpt = prime_inpt[:inpt_batch_size]

            jacs_batch.append(expected_jacobian_step(module_func, inpt_dict, "x", prime_inpt))

        exp_jac_batch = sum(jacs_batch) / len(jacs_batch)
        if apply_abs:
            exp_jac_batch = exp_jac_batch.abs()

        expected_jacs.append(exp_jac_batch.cpu() if not sum_obs else exp_jac_batch.sum(0).cpu())

    if sum_obs:
        result = torch.stack(expected_jacs).sum(0)
    else:
        result = torch.cat(expected_jacs)
    return result


def run_integrated_jacobian_scanvi(module_func, dl_base, n_steps=10, apply_abs=False, sum_obs=False):
    integrated_jacs = []

    for batch in dl_base:
        inpt_dict = batch_to_dict_scanvi(batch)
        inpt_dict["x"].requires_grad = True
#        inpt_dict["x"] = inpt_dict["x"].cuda()

        integr_jac_batch = integrated_jacobian(module_func, inpt_dict, "x", n_steps=n_steps)
        if apply_abs:
            integr_jac_batch = integr_jac_batch.abs()

        integrated_jacs.append(integr_jac_batch.cpu() if not sum_obs else integr_jac_batch.sum(0).cpu())

    if sum_obs:
        result = torch.stack(integrated_jacs).sum(0)
    else:
        result = torch.cat(integrated_jacs)
    return result
