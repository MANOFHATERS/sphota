"""DDIM sampler (the backward process) with classifier-free guidance and a guidance hook (Step 9).

At every step the network's noise estimate eps_hat gives a clean-image estimate
    x0_hat = (x_t - sqrt(1 - abar_t) eps_hat) / sqrt(abar_t)
and the next, less noisy sample is
    x_{t'} = sqrt(abar_t') x0_hat + sqrt(1 - abar_t' - sigma^2) eps_hat + sigma z,
    sigma = eta * sqrt((1 - abar_t') / (1 - abar_t)) * sqrt(1 - abar_t / abar_t').
With eta = 0 (default) the process is deterministic: the same seed always gives the same image, which the decoder
needs so that decoding a file is reproducible. 25-50 steps replace the 1000 of ancestral DDPM sampling.

Two entry points share one loop:
    ddim_sample_eps(eps_fn, sched, shape, ...)   generic: eps_fn(x_t, t) -> eps_hat (unconditional DDPM, toy models, tests)
    ddim_sample(unet, sched, c, ...)             conditional decoder with classifier-free guidance, c = condition image
``unet(x, t)`` must return an object with a ``.sample`` tensor (the diffusers UNet2DModel convention).
Samples live in [-1, 1].
"""
import torch


def _generator(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


@torch.no_grad()
def ddim_sample_eps(eps_fn, sched, shape, steps=50, eta=0.0, guide=None, seed=0, device=None, x_T=None, clip=True,
                    return_trace=False):
    """Run the reverse process.

    eps_fn(x_t, t_batch) -> eps_hat      t_batch is a LongTensor of shape (B,)
    guide(x0_hat, t) -> x0_hat           optional hook, t is the int timestep; eps_hat is re-derived from the corrected
                                         x0_hat so the update stays consistent (this is where Step 12 plugs in)
    x_T                                  optional starting noise (default: N(0, I) from ``seed``)
    clip                                 clamp x0_hat to [-1, 1] before the update (eps_hat is left as predicted)
    Returns x_0 in [-1, 1] (and, with return_trace, the list of (t, x0_hat) pairs).
    """
    device = torch.device(device) if device is not None else sched.device
    g = _generator(device, seed)
    x = torch.randn(shape, generator=g, device=device) if x_T is None else x_T.to(device).clone()
    ts = sched.ddim_timesteps(steps)
    trace = []
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        tt = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        eps = eps_fn(x, tt)
        ab = sched.ab[t].to(device)
        ab_prev = sched.ab[t_prev].to(device) if t_prev >= 0 else torch.ones((), device=device)
        x0 = sched.predict_x0(x, tt, eps)
        if clip:
            x0 = x0.clamp(-1, 1)
        if guide is not None:
            x0 = guide(x0, t)
            eps = sched.predict_eps(x, tt, x0)              # re-derive eps from the corrected x0 (as in the guide)
        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt() if t_prev >= 0 else 0.0
        dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        x = ab_prev.sqrt() * x0 + dir_xt
        if eta > 0 and t_prev >= 0:
            x = x + sigma * torch.randn(x.shape, generator=g, device=device)
        if return_trace:
            trace.append((t, x0.clone()))
    return (x, trace) if return_trace else x


@torch.no_grad()
def ddim_sample(unet, sched, c, steps=50, w=1.5, guide=None, seed=0, eta=0.0, x_T=None):
    """Conditional DDIM with classifier-free guidance: eps = e_u + w (e_c - e_u).

    c: condition image (B, C, H, W) in [-1, 1] (the coarse reconstruction); the null condition is zeros.
    w = 1 -> pure conditional (one network pass per step); w = 0 -> unconditional; larger w -> sharper but riskier.
    The network input is cat([x_t, condition], dim=1), i.e. 2C channels.
    """
    null = torch.zeros_like(c)

    def eps_fn(x, tt):
        e_c = unet(torch.cat([x, c], 1), tt).sample
        if w == 1:
            return e_c
        e_u = unet(torch.cat([x, null], 1), tt).sample
        return e_u + w * (e_c - e_u)

    return ddim_sample_eps(eps_fn, sched, c.shape, steps=steps, eta=eta, guide=guide, seed=seed, device=c.device,
                           x_T=x_T)
