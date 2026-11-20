"""Decoding with the diffusion model (Step 10): the coarse codec decode used as the condition, and full-image or tiled
diffusion decoding.

    coarse = coarse_decode(codec, x)                       # codec reconstruction from the *rounded* latent (the bitstream)
    x_hat  = diffusion_decode(unet, sched, coarse, ...)    # realistic image from the same bits

The U-Net is fully convolutional, so it decodes whole images directly. If memory is short, ``tile`` decodes
overlapping tiles that start from slices of ONE shared noise field (so neighbouring tiles agree) and blends the
overlaps with linear ramps.
"""
import torch

from ..codec.bitstream import pad_to_multiple
from .sample import ddim_sample


@torch.no_grad()
def coarse_decode(codec, x):
    """The condition: g_s(round(g_a(x))) clamped to [0, 1] - exactly the image the receiver gets from the bitstream
    without any generative step. x: (B, 3, H, W) in [0, 1], H and W multiples of 64."""
    return codec.g_s(torch.round(codec.g_a(x))).clamp(0, 1)


def tile_starts(length, tile, overlap):
    """Start offsets of tiles of size ``tile`` that overlap by ``overlap`` and cover [0, length); the last tile is
    shifted back so it ends exactly at ``length``."""
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _ramp(n, overlap, left, right, device):
    """1-D blend weights of a tile: linear ramps over ``overlap`` pixels on the sides that have a neighbouring tile."""
    w = torch.ones(n, device=device)
    if overlap > 0:
        r = (torch.arange(overlap, device=device, dtype=torch.float32) + 1) / (overlap + 1)
        if left:
            w[:overlap] = torch.minimum(w[:overlap], r)
        if right:
            w[n - overlap:] = torch.minimum(w[n - overlap:], r.flip(0))
    return w


@torch.no_grad()
def diffusion_decode(unet, sched, coarse, steps=50, w=1.5, seed=0, guide=None, tile=None, overlap=32, multiple=64,
                     eta=0.0):
    """coarse: (B, 3, H, W) in [0, 1] (any size; it is edge-padded to a multiple of ``multiple`` and cropped back).
    Returns the decoded image in [0, 1]. ``tile`` (a multiple of 8) enables tiled decoding of a padded image larger
    than one tile; a ``guide`` hook is only supported for whole-image decoding."""
    B, _, H, W = coarse.shape
    cp = pad_to_multiple(coarse, multiple)
    Hp, Wp = cp.shape[-2:]
    c = cp * 2 - 1
    g = torch.Generator(device=c.device)
    g.manual_seed(seed)
    x_T = torch.randn(c.shape, generator=g, device=c.device)               # one noise field for the whole image
    if tile is None or (Hp <= tile and Wp <= tile):
        x = ddim_sample(unet, sched, c, steps=steps, w=w, guide=guide, seed=seed, eta=eta, x_T=x_T)
    else:
        if guide is not None:
            raise NotImplementedError("guidance hooks are only supported for whole-image decoding (tile=None)")
        if tile % 8 or not 0 <= overlap < tile:
            raise ValueError("tile must be a multiple of 8 and overlap must be in [0, tile)")
        acc, den = torch.zeros_like(c), torch.zeros(1, 1, Hp, Wp, device=c.device)
        ys, xs = tile_starts(Hp, tile, overlap), tile_starts(Wp, tile, overlap)
        for y0 in ys:
            for x0 in xs:
                th, tw = min(tile, Hp), min(tile, Wp)
                sl = (slice(None), slice(None), slice(y0, y0 + th), slice(x0, x0 + tw))
                out = ddim_sample(unet, sched, c[sl], steps=steps, w=w, seed=seed, eta=eta, x_T=x_T[sl])
                wy = _ramp(th, overlap, y0 > 0, y0 + th < Hp, c.device)
                wx = _ramp(tw, overlap, x0 > 0, x0 + tw < Wp, c.device)
                wt = (wy[:, None] * wx[None, :])[None, None]
                acc[sl] += out * wt
                den[:, :, y0:y0 + th, x0:x0 + tw] += wt
        x = acc / den
    return ((x + 1) / 2).clamp(0, 1)[..., :H, :W]
