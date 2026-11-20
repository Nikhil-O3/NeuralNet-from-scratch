"""Download and decode the original MNIST IDX files without ML libraries."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATA_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename: str, destination: Path) -> None:
    """Download one public MNIST gzip file unless it already exists."""
    if not destination.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(DATA_URL + filename, destination)


def _read_images(path: Path) -> np.ndarray:
    """Decode IDX image records and normalise 0-255 pixels to floats."""
    with gzip.open(path, "rb") as source:
        magic, count, rows, columns = struct.unpack(">IIII", source.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an MNIST image file")
        pixels = np.frombuffer(source.read(), dtype=np.uint8)
    return (pixels.reshape(count, rows * columns).astype(np.float32) / 255.0)


def _read_labels(path: Path) -> np.ndarray:
    """Decode IDX label records into integer digit labels."""
    with gzip.open(path, "rb") as source:
        magic, count = struct.unpack(">II", source.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an MNIST label file")
        labels = np.frombuffer(source.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Unexpected label count in {path}")
    return labels.astype(np.int64)


def load_mnist(data_directory: str | Path = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ensure MNIST is available locally and return train/test arrays."""
    directory = Path(data_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {key: directory / filename for key, filename in FILES.items()}
    for filename, path in ((filename, paths[key]) for key, filename in FILES.items()):
        _download(filename, path)
    return _read_images(paths["train_images"]), _read_labels(paths["train_labels"]), _read_images(paths["test_images"]), _read_labels(paths["test_labels"])
