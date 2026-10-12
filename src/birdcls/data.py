"""Dataset parsing, splitting and PyTorch dataset wrappers for CUB-200-2010, free of GPU code."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

# Training filenames look like ``Black_footed_Albatross_0019_416160254.jpg`` or, for a few images without a
# Flickr id, ``Sooty_Albatross_0019_xxx.jpg``; the species name is everything before the 4-digit class index.
_TRAIN_NAME_RE = re.compile(r"^(?P<species>.+)_\d{4}_[^_]+\.jpg$")


@dataclass(frozen=True)
class Sample:
    """One annotated image; ``split`` is the image folder it lives in ("Train" or "Test")."""

    filename: str
    label: int
    split: str

    def path(self, data_root: Path) -> Path:
        """Return the image path under ``data_root``."""
        return data_root / self.split / self.filename


def read_annotations(txt_path: Path, split: str) -> list[Sample]:
    """Parse an annotation file of ``<filename> <label>`` lines into ``Sample`` records, skipping blank lines."""
    samples = []
    for line_no, line in enumerate(Path(txt_path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{txt_path}:{line_no}: expected '<filename> <label>', got {line!r}")
        samples.append(Sample(filename=parts[0], label=int(parts[1]), split=split))
    return samples


def load_splits(data_root: Path) -> tuple[list[Sample], list[Sample]]:
    """Return the official (train, test) samples of the assignment."""
    data_root = Path(data_root)
    return (
        read_annotations(data_root / "train.txt", "Train"),
        read_annotations(data_root / "test.txt", "Test"),
    )


def species_from_filename(filename: str) -> str:
    """Return the species part of a training filename."""
    match = _TRAIN_NAME_RE.match(filename)
    if match is None:
        raise ValueError(f"Cannot parse species from training filename {filename!r}")
    return match.group("species")


def derive_class_names(train: Sequence[Sample]) -> list[str]:
    """Recover the readable species name of every label from the training filenames.

    Test images are anonymized (``0000.jpg``), so the training set is the only source of names. Each label
    must map to exactly one species, so a mislabeled annotation file fails loudly.
    """
    names: dict[int, set[str]] = defaultdict(set)
    for s in train:
        names[s.label].add(species_from_filename(s.filename))
    ambiguous = {k: v for k, v in names.items() if len(v) != 1}
    if ambiguous:
        raise ValueError(f"Labels with inconsistent species names: {ambiguous}")
    num_classes = max(names) + 1
    missing = [k for k in range(num_classes) if k not in names]
    if missing:
        raise ValueError(f"Labels without any training image: {missing}")
    return [next(iter(names[k])).replace("_", " ") for k in range(num_classes)]


def stratified_holdout(
    samples: Sequence[Sample], fraction: float, seed: int, min_per_class: int = 1
) -> tuple[list[Sample], list[Sample]]:
    """Split ``samples`` into (fit, holdout) keeping every class represented in both.

    Stratifying matters because there are only 16-31 training images per class: a purely random 20% split
    could leave some species with one or zero val images and make their per-class accuracy meaningless.
    """
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)
    rng = random.Random(seed)
    fit, holdout = [], []
    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda s: s.filename)  # order-independent of input
        rng.shuffle(items)
        n_hold = max(min_per_class, round(fraction * len(items)))
        holdout.extend(items[:n_hold])
        fit.extend(items[n_hold:])
    return fit, holdout


def load_val_split(
    data_root: Path, split_json: Path
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Return (fit, val, test) using the frozen val file list in ``split_json``.

    ``scripts/prepare_data.py`` generates the split once and it is committed, so every machine and
    experiment selects hyperparameters on exactly the same images.
    """
    train, test = load_splits(data_root)
    val_names = set(json.loads(Path(split_json).read_text())["val"])
    fit = [s for s in train if s.filename not in val_names]
    val = [s for s in train if s.filename in val_names]
    if len(val) != len(val_names):
        raise ValueError("Validation split references files missing from train.txt")
    return fit, val, test


class BirdDataset(Dataset):
    """Decode a JPEG, apply ``transform`` and return (x, y, idx).

    The index lets inference code map predictions back to filenames without a second pass over the data.
    """

    def __init__(self, data_root: Path, samples: Sequence[Sample], transform: Callable | None):
        """Store the image root, the samples and the optional transform."""
        self.data_root = Path(data_root)
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return the transformed RGB image, its label and ``idx``."""
        sample = self.samples[idx]
        with Image.open(sample.path(self.data_root)) as img:
            # A handful of CUB images are grayscale or CMYK; the backbones expect 3-channel RGB.
            img = img.convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        return x, sample.label, idx


def build_train_transform(
    img_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    scale_min: float = 0.35,
    augment: str = "trivialaugment",
    random_erasing: float = 0.25,
):
    """Build the training augmentation, chosen to fight overfitting on ~24 images per class.

    * RandomResizedCrop: random zoom and aspect so the network cannot memorize exact framing. The lower
      scale bound stays well above ImageNet's 0.08 because birds often fill only part of the frame.
    * Horizontal flip: bird species are left/right symmetric; vertical flips are not natural.
    * TrivialAugmentWide / RandAugment: one or two random photometric or geometric ops per image with no
      tuned magnitude schedule, a strong default for small datasets.
    * RandomErasing: occludes a random patch, forcing the model to use several body parts (head, wing
      bars, tail) instead of a single shortcut cue.
    """
    from torchvision import transforms as T

    ops = [
        T.RandomResizedCrop(img_size, scale=(scale_min, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
    ]
    if augment == "trivialaugment":
        ops.append(T.TrivialAugmentWide(interpolation=T.InterpolationMode.BILINEAR))
    elif augment == "randaugment":
        ops.append(T.RandAugment(num_ops=2, magnitude=9, interpolation=T.InterpolationMode.BILINEAR))
    elif augment != "none":
        raise ValueError(f"Unknown augment policy {augment!r}")
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if random_erasing > 0:
        ops.append(T.RandomErasing(p=random_erasing, value="random"))
    return T.Compose(ops)


def build_eval_transform(img_size: int, mean: Sequence[float], std: Sequence[float], crop_pct: float = 1.0):
    """Build the deterministic eval preprocessing: resize the short side, then center-crop.

    ``crop_pct=1.0`` resizes the short side straight to ``img_size`` so the center crop keeps the full
    height of a landscape photo; smaller values zoom in and may clip the bird.
    """
    from torchvision import transforms as T

    return T.Compose([
        T.Resize(int(round(img_size / crop_pct)), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
