import hashlib

import numpy as np

from birdcls.fileio import read_csv, read_jsonl, save_npz_atomic, sha256_file, write_csv


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"epoch": 0}\n\n{"epoch": 1}\n')
    assert read_jsonl(path) == [{"epoch": 0}, {"epoch": 1}]


def test_csv_round_trip_keeps_column_order(tmp_path):
    path = tmp_path / "table.csv"
    write_csv(path, [{"b": 1, "a": "x"}, {"b": 2, "a": "y"}], fieldnames=["a", "b"])
    assert path.read_text().splitlines()[0] == "a,b"
    assert read_csv(path) == [{"a": "x", "b": "1"}, {"a": "y", "b": "2"}]


def test_save_npz_atomic_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "preds.npz"
    save_npz_atomic(path, logits=np.arange(4.0))
    np.testing.assert_array_equal(np.load(path)["logits"], np.arange(4.0))
    assert [p.name for p in tmp_path.iterdir()] == ["preds.npz"]


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "weights.bin"
    data = bytes(range(256)) * 5000  # larger than one read block
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()
