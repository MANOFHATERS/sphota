"""Datasets and crops."""
import os
from pathlib import Path

import numpy as np

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def read_list(path):
    """Read a file list. Relative entries are resolved against the project root."""
    with open(path) as f:
        items = [l.strip() for l in f if l.strip()]
    return [str(ROOT / p) if not Path(p).is_absolute() else p for p in items]


def default_cache_dir():
    """Where the decoded-image cache lives: $TS_IMAGE_CACHE, else <project>/data/cache. ``TS_IMAGE_CACHE=off`` disables it."""
    return os.environ.get("TS_IMAGE_CACHE") or str(ROOT / "data" / "cache")


class CropDataset(Dataset):
    """Random crops for training. Returns float tensors in [0, 1].

    If a decoded-image cache exists (truesight/image_cache.py, built by scripts/build_cache.py) the crop is cut from the cached
    uint8 pixels instead of decoding the PNG. The result is the same: a uniformly random crop (torch RNG, exactly as torchvision's
    RandomCrop draws it), a horizontal flip with probability 0.5, uint8 -> float / 255. Images missing from the cache, and images
    smaller than the crop (reflect-padded), take the original PIL path. ``cache=False`` turns the cache off."""

    def __init__(self, file_list, crop=256, cache=None):
        self.files = read_list(file_list)
        self.crop = crop
        self.tf = T.Compose([
            T.RandomCrop(crop, pad_if_needed=True, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        self.cache = None
        d = default_cache_dir() if cache in (None, True) else (None if cache is False else cache)
        if d and str(d).lower() != "off":
            from .image_cache import ImageCache
            c = ImageCache(d, ROOT)
            self.cache = c if c.usable else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if self.cache is not None:
            a = self.cache.get(self.files[i])
            if a is not None and a.shape[0] >= self.crop and a.shape[1] >= self.crop:
                h, w = a.shape[:2]
                top = torch.randint(0, h - self.crop + 1, size=(1,)).item()        # the same draws as T.RandomCrop
                left = torch.randint(0, w - self.crop + 1, size=(1,)).item()
                patch = torch.from_numpy(np.ascontiguousarray(a[top:top + self.crop, left:left + self.crop]))
                x = patch.permute(2, 0, 1).float().div_(255)
                return x.flip(-1) if torch.rand(1) < 0.5 else x                    # the same as T.RandomHorizontalFlip(0.5)
        img = Image.open(self.files[i]).convert("RGB")
        return self.tf(img)


class FullImageDataset(Dataset):
    """Whole images for testing. Pads to a multiple of 64 and remembers the size."""

    def __init__(self, file_list, multiple=64):
        self.files, self.m = read_list(file_list), multiple

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = T.ToTensor()(Image.open(self.files[i]).convert("RGB"))
        _, h, w = img.shape
        ph, pw = (-h) % self.m, (-w) % self.m
        img = torch.nn.functional.pad(img[None], (0, pw, 0, ph), mode="replicate")[0]
        return img, (h, w), self.files[i]
