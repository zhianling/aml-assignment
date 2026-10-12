"""Frozen-backbone feature extraction: embed each image and its flip once, cache them to disk."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbones import BackboneSpec
from .data import BirdDataset, Sample
from .fileio import save_npz_atomic

# Prompt template for zero-shot classification with BioCLIP; it was trained on captions mixing common and
# taxonomic names, so the common names CUB provides are a reasonable choice.
ZERO_SHOT_TEMPLATE = "a photo of {}."


class FrozenExtractor:
    """Wrap a pretrained model and expose its pooled (and, for ViTs, CLS or patch-mean) embedding."""

    def __init__(self, spec: BackboneSpec, device: torch.device, img_size: int | None = None):
        """Load the model and its preprocessing; ``img_size=None`` keeps the model's own test resolution."""
        self.spec = spec
        self.device = device
        self.tokenizer = None
        if img_size is not None and not spec.resizable:
            raise ValueError(f"{spec.key} has a fixed input resolution; img_size cannot be overridden")
        if spec.source == "timm":
            import timm

            kwargs = {}
            if img_size is not None and spec.family == "ViT":
                # ViTs size their position embeddings at construction; timm interpolates the
                # pretrained ones to the new grid. CNNs are resolution-agnostic and need nothing.
                kwargs["img_size"] = img_size
            self.model = timm.create_model(spec.name, pretrained=True, **kwargs)

            # Use the model's own eval preprocessing so every backbone is probed under its pretraining conditions;
            # the test size is used because some recipes (e.g. ResNet-50 tv2) train smaller than they test.
            cfg = timm.data.resolve_model_data_config(self.model, use_test_size=True)
            if img_size is not None:
                cfg["input_size"] = (3, img_size, img_size)
            self.transform = timm.data.create_transform(**cfg, is_training=False)
            self.img_size = int(cfg["input_size"][-1])
        elif spec.source == "open_clip":
            import open_clip

            self.model, _, self.transform = open_clip.create_model_and_transforms(spec.name)
            self.tokenizer = open_clip.get_tokenizer(spec.name)
            self.img_size = int(self.model.visual.image_size[0]
                                if isinstance(self.model.visual.image_size, (tuple, list))
                                else self.model.visual.image_size)
        else:
            raise ValueError(f"Unsupported source {spec.source!r}")
        self.model.eval().to(device)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Embed a batch; ``pooled`` is always present, ``cls`` or ``patch_mean`` is added for ViT token sequences."""
        if self.spec.source == "open_clip":
            return {"pooled": self.model.encode_image(x)}
        feats = self.model.forward_features(x)

        # ``pre_logits=True`` returns exactly the vector the model's own classifier consumes (CLS token for
        # most ViTs, mean patch token for DINOv3 and EVA-02 MIM, global average pool for CNNs, attention pool
        # for SigLIP).
        out = {"pooled": self.model.forward_head(feats, pre_logits=True)}
        if feats.ndim == 3:
            # ViT token sequence [B, prefix + patches, D]. DINOv2's linear-evaluation protocol concatenates the
            # CLS token with the mean patch token, so whichever of the two the model's own head does not
            # already use is stored alongside ``pooled``.
            n_prefix = getattr(self.model, "num_prefix_tokens", 1)
            if getattr(self.model, "global_pool", "token") == "avg":
                if n_prefix >= 1:
                    # Average-pooling heads never read the CLS token, although DINO-style pretraining puts
                    # its image-level objective exactly there.
                    out["cls"] = feats[:, 0]
            elif feats.shape[1] > n_prefix:
                out["patch_mean"] = feats[:, n_prefix:].mean(dim=1)
        return out

    @torch.no_grad()
    def text_embeddings(self, class_names: Sequence[str]) -> torch.Tensor | None:
        """Embed the zero-shot prompt of every class name, or return None for an image-only model."""
        if self.tokenizer is None:
            return None
        tokens = self.tokenizer([ZERO_SHOT_TEMPLATE.format(n) for n in class_names]).to(self.device)
        return self.model.encode_text(tokens)


def _embed_with_fallback(extractor: FrozenExtractor, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Embed in half precision, which roughly doubles MPS/CUDA throughput, and fall back to fp32 on overflow."""
    if extractor.device.type in ("mps", "cuda"):
        with torch.autocast(device_type=extractor.device.type, dtype=torch.float16):
            out = extractor.embed(x)
        if all(torch.isfinite(v).all() for v in out.values()):
            return {k: v.float() for k, v in out.items()}
    return {k: v.float() for k, v in extractor.embed(x).items()}


def extract_to_file(
    spec: BackboneSpec,
    samples: Sequence[Sample],
    data_root: Path,
    class_names: Sequence[str],
    out_path: Path,
    device: torch.device,
    img_size: int | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Embed every sample (original and flipped view), save an uncompressed fp16 .npz and return its metadata."""
    # Build the extractor and loader
    extractor = FrozenExtractor(spec, device, img_size)
    loader = DataLoader(
        BirdDataset(data_root, samples, extractor.transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # Embed both views of every batch
    chunks: dict[str, list[np.ndarray]] = {}
    start = time.perf_counter()
    for batch_idx, (x, _, _) in enumerate(loader):
        if batch_idx and batch_idx % 50 == 0:
            done = batch_idx * batch_size
            print(f"  {spec.key}: {done}/{len(samples)} images, "
                  f"{2 * done / (time.perf_counter() - start):.1f} views/s", flush=True)
        x = x.to(device, non_blocking=True)
        views = [_embed_with_fallback(extractor, x), _embed_with_fallback(extractor, torch.flip(x, dims=[3]))]
        for name in views[0]:
            stacked = torch.stack([v[name] for v in views], dim=1)  # [B, 2 views, D]
            chunks.setdefault(name, []).append(stacked.cpu().numpy().astype(np.float16))
    elapsed = time.perf_counter() - start

    # Collect the arrays and metadata
    arrays = {f"feat_{k}": np.concatenate(v) for k, v in chunks.items()}
    # The probe uses only the image network, so an image-text model's text encoder is not counted.
    image_net = extractor.model.visual if spec.source == "open_clip" else extractor.model
    text = extractor.text_embeddings(class_names)
    if text is not None:
        arrays["text"] = text.float().cpu().numpy()
    meta = {
        "key": spec.key, "name": spec.name, "family": spec.family, "pretraining": spec.pretraining,
        "in_domain": spec.in_domain, "img_size": extractor.img_size, "num_images": len(samples),
        "seconds": round(elapsed, 1), "images_per_second": round(2 * len(samples) / elapsed, 1),
        "device": str(device), "params_m": round(sum(p.numel() for p in image_net.parameters()) / 1e6, 1),
    }

    # Save the feature file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(
        out_path,
        labels=np.array([s.label for s in samples], dtype=np.int16),
        filenames=np.array([s.filename for s in samples]),
        splits=np.array([s.split for s in samples]),
        meta=np.array(json.dumps(meta)),
        **arrays,
    )
    return meta


def load_features(path: Path) -> dict:
    """Load a cached feature file into plain numpy arrays plus its ``meta`` dict."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["meta"] = json.loads(str(data["meta"]))
    return data
