"""Evaluation of trained TrueSight codecs (Step 7): estimated-bpp rate-distortion points, BD-rate, K ablation.

bpp here is the *estimated* rate = -sum log2 p / (original H*W) from the entropy model (an ideal arithmetic coder).
Real bitstreams and the header cost come in Step 8. Images are padded to a multiple of 64 and cropped back; the
padded area's bits are charged to the image (honest: they are in the file).
"""
import math

import numpy as np
import pandas as pd
import torch

from ..data import FullImageDataset
from ..metrics import bd_rate, lpips_dist, msssim, psnr
from .vae_codec import TrueSightCodec, bpp_terms


@torch.no_grad()
def evaluate_codec(net, files_list, device=None, max_images=None, with_perceptual=True):
    """One row per image: bpp (total / y / z), PSNR, MS-SSIM, LPIPS. ``files_list`` is a split-list .txt path."""
    device = torch.device(device or next(net.parameters()).device)
    net.eval()
    ds = FullImageDataset(files_list, multiple=TrueSightCodec.DOWN)
    rows = []
    for i in range(len(ds) if max_images is None else min(max_images, len(ds))):
        x, (h, w), path = ds[i]
        x = x[None].to(device)
        out = net(x)
        terms = bpp_terms(out["likelihoods"], h * w)                 # normalise by ORIGINAL pixels
        xr = out["x_hat"][..., :h, :w].clamp(0, 1).cpu()
        xo = x[..., :h, :w].cpu()
        row = dict(image=path.split("/")[-1].split("\\")[-1], bpp=sum(terms.values()).item(), bpp_y=terms["y"].item(),
                   bpp_z=terms["z"].item(), psnr=psnr(xo, xr))
        if with_perceptual:
            row.update(msssim=msssim(xo, xr), lpips=lpips_dist(xo, xr))
        rows.append(row)
    return pd.DataFrame(rows)


def mean_point(df, **extra):
    """Dataset-level RD point = mean over images (the convention used for the JPEG/WebP/PCA/VQ curves)."""
    return {**extra, **{c: float(df[c].mean()) for c in df.columns if c != "image"}}


def beats_baseline(codec_pts, base_pts, max_bpp=0.5):
    """For every codec point with bpp < max_bpp inside the baseline's bpp range, is codec PSNR above the baseline's
    PSNR interpolated (linearly in log-bpp) at that bpp? Returns (all_ok, details list)."""
    b = base_pts.sort_values("bpp")
    lb, bp = np.log(b.bpp.to_numpy()), b.psnr.to_numpy()
    details = []
    for _, r in codec_pts.sort_values("bpp").iterrows():
        if r.bpp >= max_bpp:
            continue
        if not (lb.min() <= math.log(r.bpp) <= lb.max()):
            details.append(dict(bpp=r.bpp, psnr=r.psnr, baseline_psnr=float("nan"), ok=None, note="outside baseline range"))
            continue
        ref = float(np.interp(math.log(r.bpp), lb, bp))
        details.append(dict(bpp=r.bpp, psnr=r.psnr, baseline_psnr=ref, ok=bool(r.psnr > ref), note=""))
    checked = [d for d in details if d["ok"] is not None]
    return (bool(checked) and all(d["ok"] for d in checked)), details


def bd_table(codec_pts, baselines, metric="psnr"):
    """BD-rate (%) of the codec vs each baseline ({name: DataFrame with bpp, <metric>}); negative = codec saves bits.
    NaN where a baseline has < 4 points or the quality ranges do not overlap."""
    out = {}
    for name, b in baselines.items():
        try:
            out[name] = bd_rate(b.bpp, b[metric], codec_pts.bpp, codec_pts[metric])
        except ValueError:
            out[name] = float("nan")
    return out


def ablation_k(k3, k1):
    """K=3 vs K=1 at the lambdas both were trained for. k3 has all lambdas (>= 2 points) and k1 a subset; both are
    DataFrames with columns lmbda, bpp, psnr. For each K=1 point: the bpp the K=3 curve needs for the same PSNR
    (interpolated in log-bpp) and the saving in %. Also the raw (bpp, PSNR) pairs at identical lambda."""
    k3 = k3.sort_values("psnr")
    rows = []
    for _, r1 in k1.iterrows():
        r3 = k3[np.isclose(k3.lmbda, r1.lmbda)]
        raw = {}
        if len(r3):
            raw = dict(bpp_K3=float(r3.bpp.iloc[0]), psnr_K3=float(r3.psnr.iloc[0]),
                       d_bpp_pct_same_lambda=(float(r3.bpp.iloc[0]) / r1.bpp - 1) * 100,
                       d_psnr_same_lambda=float(r3.psnr.iloc[0]) - r1.psnr)
        lo, hi = k3.psnr.min(), k3.psnr.max()
        if len(k3) >= 2 and lo <= r1.psnr <= hi:
            bpp3 = math.exp(float(np.interp(r1.psnr, k3.psnr.to_numpy(), np.log(k3.bpp.to_numpy()))))
            eq = dict(bpp_K3_at_equal_psnr=bpp3, saving_pct_equal_psnr=(bpp3 / r1.bpp - 1) * 100)
        else:
            eq = dict(bpp_K3_at_equal_psnr=float("nan"), saving_pct_equal_psnr=float("nan"))
        rows.append(dict(lmbda=r1.lmbda, bpp_K1=r1.bpp, psnr_K1=r1.psnr, **raw, **eq))
    return pd.DataFrame(rows)
