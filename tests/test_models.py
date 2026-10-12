import pytest
import torch

from birdcls.config import TrainConfig
from birdcls.models import NUM_CLASSES, build_model


def _save_probe(path, variant, dim=192, fitted_on="fit"):
    """Save a fake probe state to ``path`` and return it."""
    torch.manual_seed(0)
    state = {"weight": torch.randn(NUM_CLASSES, dim), "bias": torch.randn(NUM_CLASSES),
             "variant": variant, "weight_decay": 1e-4, "fitted_on": fitted_on, "meta": {"img_size": 64}}
    torch.save(state, path)
    return state


def _cfg(head, global_pool, use_val=True):
    """Build a tiny untrained ViT config whose head starts from the probe at ``head``."""
    return TrainConfig(run_name="t", backbone="vit_tiny_patch16_224", pretrained=False, img_size=64,
                       head_init=str(head), global_pool=global_pool, use_val=use_val)


def test_cls_probe_initialises_a_token_pooled_head(tmp_path):
    state = _save_probe(tmp_path / "cls.pt", "cls")
    model, _ = build_model(_cfg(tmp_path / "cls.pt", "token"))
    torch.testing.assert_close(model.get_classifier().weight, state["weight"])
    x = torch.randn(1, 3, 224, 224)
    feats = model.eval().forward_features(x)
    torch.testing.assert_close(model.forward_head(feats, pre_logits=True), feats[:, 0])


@pytest.mark.parametrize("variant, global_pool", [("pooled", "token"), ("cls", "")])
def test_mismatched_probe_variant_is_refused(tmp_path, variant, global_pool):
    _save_probe(tmp_path / "head.pt", variant)
    with pytest.raises(ValueError, match="classifier reads"):
        build_model(_cfg(tmp_path / "head.pt", global_pool))


@pytest.mark.parametrize("fitted_on", ["train", None])
def test_a_head_that_saw_val_is_refused_when_the_run_selects_on_val(tmp_path, fitted_on):
    state = _save_probe(tmp_path / "head.pt", "pooled", fitted_on=fitted_on)
    if fitted_on is None:  # heads saved before the field existed were refit on all training images
        del state["fitted_on"]
        torch.save(state, tmp_path / "head.pt")
    with pytest.raises(ValueError, match="selects on val"):
        build_model(_cfg(tmp_path / "head.pt", ""))
    model, _ = build_model(_cfg(tmp_path / "head.pt", "", use_val=False))  # trains on everything: fine
    torch.testing.assert_close(model.get_classifier().weight, state["weight"])


def test_unsupported_pooling_is_refused(tmp_path):
    with pytest.raises(ValueError, match="global_pool"):
        build_model(_cfg("", "avg"))


def test_head_dropout_reaches_the_classifier_and_is_off_by_default():
    def head_drop(drop_rate):
        """Build a tiny ViT with ``drop_rate`` and return the probability of its head dropout."""
        model, _ = build_model(TrainConfig(run_name="t", backbone="vit_tiny_patch16_224", pretrained=False,
                                           img_size=64, drop_rate=drop_rate))
        return model.head_drop.p

    assert head_drop(0.3) == 0.3 and TrainConfig(run_name="t", backbone="x").drop_rate == 0.0
