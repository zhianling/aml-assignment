import pytest
import timm
import torch

from birdcls.backbones import BackboneSpec
from birdcls.features import FrozenExtractor


def extractor_for(model: torch.nn.Module, family: str) -> FrozenExtractor:
    """Build a ``FrozenExtractor`` around ``model`` without running ``__init__``."""
    # __init__ downloads pretrained weights; embed() only needs the spec and the model.
    extractor = FrozenExtractor.__new__(FrozenExtractor)
    extractor.spec = BackboneSpec("test", "timm", "test", family, "none")
    extractor.model = model.eval()
    return extractor


@pytest.mark.parametrize("global_pool, expected", [
    ("avg", {"pooled", "cls"}),            # DINOv3-style: head averages patches, CLS stored separately
    ("token", {"pooled", "patch_mean"}),   # DINOv2-style: head reads CLS, patch mean stored separately
])
def test_vit_caches_the_summary_its_head_ignores(global_pool, expected):
    # Embed a batch with a tiny ViT
    torch.manual_seed(0)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=64, reg_tokens=4,
                              global_pool=global_pool, num_classes=0)
    x = torch.randn(2, 3, 64, 64)
    out = extractor_for(model, "ViT").embed(x)
    assert set(out) == expected

    # Check the cached summaries against the token outputs
    tokens = model.forward_features(x)
    if global_pool == "avg":
        torch.testing.assert_close(out["cls"], tokens[:, 0])
    else:
        torch.testing.assert_close(out["pooled"], tokens[:, 0])
        # Register tokens are prefix tokens too and must not leak into the patch average.
        torch.testing.assert_close(out["patch_mean"], tokens[:, model.num_prefix_tokens:].mean(dim=1))


def test_cnn_caches_only_the_pooled_vector():
    model = timm.create_model("resnet18", pretrained=False, num_classes=0)
    out = extractor_for(model, "CNN").embed(torch.randn(2, 3, 64, 64))
    assert set(out) == {"pooled"} and out["pooled"].shape == (2, 512)
