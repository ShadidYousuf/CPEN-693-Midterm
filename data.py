import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent / "data"
PROBLEMS = {1: "cifar10", 2: "cifar10", 3: "fashion", 4: "flowers"}
INFO = {
    "cifar10": {"channels": 3, "size": 32, "names": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]},
    "fashion": {"channels": 1, "size": 28, "names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]},
    "flowers": {"channels": 3, "size": 64, "names": [f"class {i}" for i in range(102)]},
}


def split_train_val(images, labels, split):
    order = np.random.RandomState(0).permutation(len(labels))
    chosen = order[:5000] if split == "val" else order[5000:]
    return images[chosen], labels[chosen]


def load_cifar10(split):
    files = ["test_batch"] if split == "test" else [f"data_batch_{i}" for i in range(1, 6)]
    images, labels = [], []
    for file in files:
        with open(ROOT / "cifar-10-batches-py" / file, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        images.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels += batch[b"labels"]
    images, labels = np.concatenate(images), np.array(labels)
    return (images, labels) if split == "test" else split_train_val(images, labels, split)


def load_fashion(split):
    prefix = "t10k" if split == "test" else "train"
    with open(ROOT / "FashionMNIST" / "raw" / f"{prefix}-images-idx3-ubyte", "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with open(ROOT / "FashionMNIST" / "raw" / f"{prefix}-labels-idx1-ubyte", "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return (images, labels) if split == "test" else split_train_val(images, labels, split)


def flower_files(split):
    key = {"train": "trnid", "val": "valid", "test": "tstid"}[split]
    ids = loadmat(ROOT / "flowers-102" / "setid.mat")[key][0]
    labels = loadmat(ROOT / "flowers-102" / "imagelabels.mat")["labels"][0][ids - 1] - 1
    return [ROOT / "flowers-102" / "jpg" / f"image_{i:05d}.jpg" for i in ids], labels


def load_flowers(split):
    size = INFO["flowers"]["size"]
    paths, labels = flower_files(split)
    images = [np.array(Image.open(path).convert("RGB").resize((size, size))) for path in paths]
    return np.stack(images), labels


LOADERS = {"cifar10": load_cifar10, "fashion": load_fashion, "flowers": load_flowers}


def shift_flip(images):
    n, size = images.shape[0], images.shape[-1]
    theta = torch.zeros(n, 2, 3, device=images.device)
    theta[:, 0, 0] = torch.where(torch.rand(n, device=images.device) < 0.5, -1.0, 1.0)
    theta[:, 1, 1] = 1
    theta[:, :, 2] = torch.randint(-4, 5, (n, 2), device=images.device) * 2 / size
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def zoom_rotate_flip(images):
    n, device = images.shape[0], images.device
    angle = (torch.rand(n, device=device) * 2 - 1) * 0.52
    zoom = 0.6 + 0.4 * torch.rand(n, device=device)
    flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = zoom * torch.cos(angle) * flip
    theta[:, 0, 1] = -zoom * torch.sin(angle)
    theta[:, 1, 0] = zoom * torch.sin(angle) * flip
    theta[:, 1, 1] = zoom * torch.cos(angle)
    theta[:, :, 2] = (torch.rand(n, 2, device=device) * 2 - 1) * (1 - zoom[:, None])
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, padding_mode="reflection", align_corners=False)


def change_brightness_contrast(images):
    n, device = images.shape[0], images.device
    brightness = 0.7 + 0.6 * torch.rand(n, 1, 1, 1, device=device)
    contrast = 0.7 + 0.6 * torch.rand(n, 1, 1, 1, device=device)
    mean = images.mean((1, 2, 3), keepdim=True)
    return (((images - mean) * contrast + mean) * brightness).clamp(0, 1)


def flower_augment(images):
    return change_brightness_contrast(zoom_rotate_flip(images))


AUGMENTS = {"cifar10": shift_flip, "fashion": shift_flip, "flowers": flower_augment}


class ImageDataset(Dataset):
    def __init__(self, images, labels, augment=None):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def get_dataset(name, split):
    images, labels = LOADERS[name](split)
    images = torch.tensor(images)
    images = images[:, None] if images.ndim == 3 else images.permute(0, 3, 1, 2).contiguous()
    augment = AUGMENTS[name] if split == "train" else None
    return ImageDataset(images, torch.tensor(labels, dtype=torch.long), augment)


def prepare(images, augment=None):
    images = images.float() / 255
    if augment:
        images = augment(images)
    return (images - 0.5) / 0.5


def get_loader(name, split, batch_size):
    return DataLoader(get_dataset(name, split), batch_size=batch_size, shuffle=split == "train")
