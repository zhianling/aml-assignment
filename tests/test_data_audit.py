import json

import numpy as np
import pytest
from PIL import Image

from birdcls.audit import (dhash, flagged_test_images, group_exact_duplicates, near_duplicate_pairs,
                           nearest_neighbours)
from birdcls.data import Sample, derive_class_names, read_annotations, species_from_filename, stratified_holdout


def test_species_from_filename_handles_flickr_id_and_placeholder():
    assert species_from_filename("Black_footed_Albatross_0019_416160254.jpg") == "Black_footed_Albatross"
    assert species_from_filename("Sooty_Albatross_0019_xxx.jpg") == "Sooty_Albatross"


def test_read_annotations_rejects_malformed_line(tmp_path):
    p = tmp_path / "train.txt"
    p.write_text("a_0001_1.jpg 0\n\nbroken line here\n")
    with pytest.raises(ValueError):
        read_annotations(p, "Train")


def test_derive_class_names_rejects_label_with_two_species():
    samples = [Sample("A_bird_0001_1.jpg", 0, "Train"), Sample("B_bird_0001_2.jpg", 0, "Train")]
    with pytest.raises(ValueError):
        derive_class_names(samples)


def test_stratified_holdout_keeps_every_class_and_is_deterministic():
    samples = [Sample(f"S{c}_{i:04d}_x.jpg", c, "Train") for c in range(5) for i in range(16 + c)]
    fit1, val1 = stratified_holdout(samples, 0.2, seed=1, min_per_class=3)
    fit2, val2 = stratified_holdout(list(reversed(samples)), 0.2, seed=1, min_per_class=3)
    assert {s.label for s in val1} == set(range(5))
    assert all(sum(s.label == c for s in val1) >= 3 for c in range(5))
    assert {s.filename for s in val1} == {s.filename for s in val2}  # input order must not matter
    assert len(fit1) + len(val1) == len(samples)


def test_dhash_is_stable_under_resize_and_distinguishes_images():
    img = Image.effect_mandelbrot((256, 192), (-2, -1.5, 1, 1.5), 50).convert("RGB")
    # The set is symmetric top to bottom only, so a left-right flip gives a different image.
    other = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    h, h_small, h_other = dhash(img), dhash(img.resize((128, 96))), dhash(other)
    assert bin(h ^ h_small).count("1") <= 4
    assert bin(h ^ h_other).count("1") > 10


def test_duplicate_grouping_and_pairs():
    assert group_exact_duplicates(["a", "b", "a", "c", "b"]) == [[0, 2], [1, 4]]
    pairs = near_duplicate_pairs([0b0000, 0b0001, 0b1111_0000], max_distance=1)
    assert pairs == [(0, 1, 1)]


def test_nearest_neighbours_uses_cosine_not_distance():
    reference = np.array([[1.0, 0.0], [10.0, 10.0]])
    idx, cos = nearest_neighbours(np.array([[0.1, 0.1], [3.0, 0.0]]), reference)
    assert idx.tolist() == [1, 0]
    np.testing.assert_allclose(cos, [1.0, 1.0])


def _record(name, split, label):
    """Build an audit record for one image."""
    return {"filename": name, "split": split, "label": label}


def test_flagged_test_images_combines_hash_and_embedding_evidence(tmp_path):
    # Write the hash audit, the review and the embedding audit
    (tmp_path / "audit.json").write_text(json.dumps({
        "exact_duplicates": {"groups": [
            [_record("a.jpg", "Train", 1), _record("t1.jpg", "Test", 2)],  # train/test, labels differ
            [_record("b.jpg", "Train", 1), _record("c.jpg", "Train", 1)],  # within train: ignored
        ]},
        "near_duplicates": {"pairs": [
            {"a": _record("d.jpg", "Train", 3), "b": _record("t2.jpg", "Test", 3), "distance": 4},
            {"a": _record("e.jpg", "Train", 4), "b": _record("t3.jpg", "Test", 4), "distance": 5},
        ]},
    }))
    review = {"false_positive_pairs": [["e.jpg", "t3.jpg"]], "same_session_threshold": 0.98}
    (tmp_path / "dedup_review.json").write_text(json.dumps(review))
    (tmp_path / "embedding_audit.json").write_text(json.dumps({"test": [
        {"file": "t1.jpg", "nearest_train": "a.jpg", "cosine": 1.0, "same_label": False},
        {"file": "t4.jpg", "nearest_train": "f.jpg", "cosine": 0.99, "same_label": True},
        {"file": "t5.jpg", "nearest_train": "g.jpg", "cosine": 0.97, "same_label": True},
    ]}))

    # The stronger hash evidence keeps its reason; the reviewed false positive and the below-threshold neighbor
    # stay in the clean subset.
    flags = flagged_test_images(tmp_path)
    assert flags == {"t1.jpg": "exact-conflict", "t2.jpg": "near-same-label", "t4.jpg": "same-session"}

    # Without a threshold in the review, t4 is no longer flagged.
    del review["same_session_threshold"]
    (tmp_path / "dedup_review.json").write_text(json.dumps(review))
    assert "t4.jpg" not in flagged_test_images(tmp_path)
