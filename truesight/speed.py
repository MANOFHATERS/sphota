"""Speed settings that change how fast a GPU computes, not what it computes (up to float rounding noise).

    cudnn.benchmark   cuDNN times the convolution algorithms once per input shape and keeps the fastest. Training uses fixed
                      shapes, so the one-off search is paid back. On by default on CUDA.            TS_CUDNN_BENCHMARK=0 turns it off
    channels_last     NHWC memory layout; with fp16 autocast the Tensor Cores of a T4 run convolutions faster in it. PyTorch advises
                      pairing it with AMP, so it is on for the diffusion U-Net (fp16) and OFF for the fp32 codec by default.
                                                                                              TS_CHANNELS_LAST = none | unet | all
    fused EMA         one ``torch._foreach`` call instead of a Python loop over every parameter (in train_diffusion.update_ema).

None of these touch the data, the model, the loss, the optimiser or the iteration counts. ``scripts/profile_train.py`` measures each
setting on your GPU so you can keep only what helps (turn one off with the variables above).
"""
import os

import torch

CHANNELS_LAST_MODES = ("none", "unet", "all")


def _flag(name, default):
    v = os.environ.get(name)
    return default if v is None or not v.strip() else v.strip().lower() not in ("0", "false", "off", "no")


def cudnn_benchmark_enabled():
    return _flag("TS_CUDNN_BENCHMARK", True)


def channels_last_mode():
    m = os.environ.get("TS_CHANNELS_LAST", "unet").strip().lower() or "unet"
    if m not in CHANNELS_LAST_MODES:
        raise ValueError(f"TS_CHANNELS_LAST must be one of {CHANNELS_LAST_MODES}, got {m!r}")
    return m


def setup(device):
    """Process-wide settings for a trainer on ``device``. Does nothing on the CPU."""
    device = torch.device(device)
    if device.type == "cuda" and cudnn_benchmark_enabled():
        torch.backends.cudnn.benchmark = True
    return device


def use_channels_last(kind, device, amp=True):
    """Should the ``kind`` model ("unet" or "codec") run in channels_last on this device?"""
    if torch.device(device).type != "cuda":
        return False
    mode = channels_last_mode()
    return (mode == "all") or (mode == "unet" and kind == "unet" and amp)


def to_channels_last(*modules):
    for m in modules:
        m.to(memory_format=torch.channels_last)


def cl(x, on):
    """``x`` in channels_last layout when ``on`` (same values, different strides)."""
    return x.contiguous(memory_format=torch.channels_last) if on else x
