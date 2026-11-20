"""Datasets and crops."""
from pathlib import Path

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


class CropDataset(Dataset):
    """Random crops for training. Returns float tensors in [0, 1]."""

    def __init__(self, file_list, crop=256):
        self.files = read_list(file_list)
        self.tf = T.Compose([
            T.RandomCrop(crop, pad_if_needed=True, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
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
