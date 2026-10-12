"""Registry of the pretrained backbones compared in this project, keyed by short names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    """Description of one pretrained backbone: where to load it from and how it was pretrained."""

    key: str          # short identifier used in file names and tables
    source: str       # "timm" (incl. hf-hub:timm/...) or "open_clip"
    name: str         # model identifier understood by that library
    family: str       # "CNN" or "ViT" (Swin counted as ViT: attention-based)
    pretraining: str  # one-line description for report tables
    in_domain: bool = False  # True when pretraining already saw bird-species labels
    # False when the input resolution is baked into the weights: Swin's attention windows
    # must tile the feature map, and open_clip towers have fixed position embeddings.
    resizable: bool = True


BACKBONES: dict[str, BackboneSpec] = {
    spec.key: spec
    for spec in [
        # CNNs
        BackboneSpec("resnet50_in1k", "timm", "resnet50.tv2_in1k", "CNN",
                     "ImageNet-1k supervised (classic baseline)"),
        BackboneSpec("convnext_l_in22k", "timm", "convnext_large.fb_in22k_ft_in1k_384", "CNN",
                     "ImageNet-22k supervised"),
        BackboneSpec("convnext_l_clip", "timm", "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384", "CNN",
                     "CLIP on LAION-2B, then ImageNet-12k/1k"),
        BackboneSpec("convnext_l_dinov3", "timm", "convnext_large.dinov3_lvd1689m", "CNN",
                     "DINOv3 self-supervised (LVD-1.7B), distilled from ViT-7B"),
        BackboneSpec("convnext_l_inat21", "timm", "hf-hub:timm/convnext_large_mlp.laion2b_ft_augreg_inat21", "CNN",
                     "CLIP on LAION-2B, then iNaturalist-2021 species", in_domain=True),
        # Vision Transformers
        BackboneSpec("swin_l_in22k", "timm", "swin_large_patch4_window12_384.ms_in22k", "ViT",
                     "ImageNet-22k supervised (Swin, as in HERBS)", resizable=False),
        BackboneSpec("eva02_l_m38m", "timm", "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", "ViT",
                     "EVA-02 masked image modelling (Merged-38M), then ImageNet-22k/1k"),
        BackboneSpec("siglip2_so400m", "timm", "vit_so400m_patch16_siglip_384.v2_webli", "ViT",
                     "SigLIP 2 image-text contrastive (WebLI)"),
        BackboneSpec("vit_l_dinov2", "timm", "vit_large_patch14_reg4_dinov2.lvd142m", "ViT",
                     "DINOv2 self-supervised (LVD-142M)"),
        BackboneSpec("vit_l_dinov3", "timm", "vit_large_patch16_dinov3.lvd1689m", "ViT",
                     "DINOv3 self-supervised (LVD-1.7B)"),
        BackboneSpec("vit_hp_dinov3", "timm", "vit_huge_plus_patch16_dinov3.lvd1689m", "ViT",
                     "DINOv3 self-supervised (LVD-1.7B), ViT-H+"),
        BackboneSpec("eva02_l_inat21", "timm", "hf-hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21", "ViT",
                     "EVA-02 CLIP (Merged-2B), then iNaturalist-2021 species", in_domain=True),
        BackboneSpec("bioclip2", "open_clip", "hf-hub:imageomics/bioclip-2", "ViT",
                     "BioCLIP 2: CLIP on TreeOfLife-200M taxonomic captions", in_domain=True, resizable=False),
        BackboneSpec("bioclip25_h", "open_clip", "hf-hub:imageomics/bioclip-2.5-vith14", "ViT",
                     "BioCLIP 2.5: CLIP on TreeOfLife taxonomic captions, ViT-H/14", in_domain=True,
                     resizable=False),
    ]
}


def get_backbone(key: str) -> BackboneSpec:
    """Return the spec registered under ``key``, or raise a KeyError that lists the valid keys."""
    try:
        return BACKBONES[key]
    except KeyError:
        raise KeyError(f"Unknown backbone {key!r}; choose from {sorted(BACKBONES)}") from None
