"""Dataset loading for the benchmark reproduction. MNIST and Fashion-MNIST
come from torchvision; ISOLET isn't in torchvision/torchtext, so it's
fetched once from the UCI ML Repository and cached locally."""

import zipfile
from pathlib import Path

import requests
import torch
import torchvision
import unlzw3

CACHE_DIR = Path(__file__).parent / "data_cache"
ISOLET_URL = "https://archive.ics.uci.edu/static/public/54/isolet.zip"


def load_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(str(CACHE_DIR), train=train, download=True)
    X = ds.data.reshape(len(ds), -1).float() / 255.0
    return X, ds.targets


def load_fashion_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.FashionMNIST(str(CACHE_DIR), train=train, download=True)
    X = ds.data.reshape(len(ds), -1).float() / 255.0
    return X, ds.targets


def _parse_isolet_file(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [line.strip().split(",") for line in text.strip().splitlines() if line.strip()]
    X = torch.tensor([[float(v) for v in row[:-1]] for row in rows])
    y = torch.tensor([int(float(row[-1])) - 1 for row in rows])  # labels are 1..26
    return X, y


def load_isolet(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "isolet.zip"
    if not zip_path.exists():
        response = requests.get(ISOLET_URL, timeout=60)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path) as zf:
        name = "isolet1+2+3+4.data.Z" if train else "isolet5.data.Z"
        with zf.open(name) as f:
            raw = f.read()
    text = unlzw3.unlzw(raw).decode("utf-8")
    return _parse_isolet_file(text)
