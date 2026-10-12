import argparse
import time
import zipfile
from pathlib import Path

import pytest

from script_loader import load_script

remote_job = load_script("remote_job")


def _write_source(src, zipped: bool, n_test: int = 2):
    """Write a fake dataset source folder, zipped or unpacked, with ``n_test`` test images."""
    src.mkdir(parents=True)
    (src / "train.txt").write_text("a.jpg 0\nb.jpg 1\n")
    (src / "test.txt").write_text("".join(f"{i:04d}.jpg 0\n" for i in range(2)))
    files = {"Train": ["a.jpg", "b.jpg"], "Test": [f"{i:04d}.jpg" for i in range(n_test)]}
    for folder, names in files.items():
        if zipped:
            with zipfile.ZipFile(src / f"{folder}.zip", "w") as zf:
                for n in names:
                    zf.writestr(f"{folder}/{n}", b"jpg")
        else:
            # Kaggle may unpack an uploaded archive into a nested folder of the same name.
            (src / folder / folder).mkdir(parents=True)
            for n in names:
                (src / folder / folder / n).write_bytes(b"jpg")


@pytest.mark.parametrize("zipped", [True, False])
def test_stage_data_from_zips_or_unpacked_folders(tmp_path, zipped):
    _write_source(tmp_path / "src", zipped)
    remote_job.stage_data(tmp_path / "src", tmp_path / "dest")
    assert sorted(p.name for p in (tmp_path / "dest" / "Train").iterdir()) == ["a.jpg", "b.jpg"]
    assert (tmp_path / "dest" / "test.txt").exists()
    assert not list((tmp_path / "dest").glob(".*partial"))
    remote_job.stage_data(tmp_path / "src", tmp_path / "dest")  # already staged: a no-op


def test_stage_data_rejects_incomplete_archives(tmp_path):
    _write_source(tmp_path / "src", zipped=True, n_test=1)
    with pytest.raises(SystemExit, match="Test: 1 images"):
        remote_job.stage_data(tmp_path / "src", tmp_path / "dest")


def _lane_args(**overrides):
    """Build the parsed arguments of a lane, with ``overrides`` replacing the defaults."""
    base = dict(overrides=["num_workers=2"], session_hours=0.0, benchmark_only=False, run_root=Path("/runs"))
    return argparse.Namespace(**{**base, **overrides})


def test_lane_passes_the_remaining_session_time_and_skips_when_too_little_is_left(monkeypatch):
    # Run a lane that started an hour ago in a 9 hour session
    calls = []
    monkeypatch.setattr(remote_job, "run_streaming", lambda cmd, env, prefix: calls.append((cmd, env)) or 0)
    now = time.monotonic()
    failed = []
    remote_job.run_lane(1, [("train", "a.toml")], _lane_args(session_hours=9.0), now - 3600, failed)

    # Check the time budget is what remains of the session
    cmd, env = calls[0]
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and "num_workers=2" in cmd
    budget = float(cmd[cmd.index("--set", cmd.index("num_workers=2")) + 1].split("=")[1])
    assert 7.99 < budget <= 8.0

    # Skip the next config when only 0.2 hours remain
    calls.clear()
    remote_job.run_lane(None, [("train", "b.toml")], _lane_args(session_hours=1.2), now - 3600, failed)
    assert calls == [] and failed == []


def test_lane_records_failed_configs_and_keeps_going(monkeypatch):
    monkeypatch.setattr(remote_job, "run_streaming", lambda cmd, env, prefix: int("bad" in cmd[-1]))
    failed = []
    remote_job.run_lane(None, [("train", "bad.toml"), ("train", "good.toml")], _lane_args(overrides=[]),
                        time.monotonic(), failed)
    assert failed == ["bad.toml"]


def test_zoom_tasks_score_finished_runs_once(tmp_path, monkeypatch):
    # Make a run with raw arrays and a run with only an old report
    calls = []
    monkeypatch.setattr(remote_job, "run_streaming", lambda cmd, env, prefix: calls.append(cmd) or 0)
    (tmp_path / "scored").mkdir()
    (tmp_path / "scored" / "zoom_raw.npz").write_bytes(b"")
    (tmp_path / "old_report").mkdir()
    (tmp_path / "old_report" / "zoom.json").write_text("{}")  # scored before the raw arrays were kept

    # Run the lane with those two and a fresh run
    remote_job.run_lane(None, [("zoom", "scored"), ("zoom", "old_report"), ("zoom", "fresh")],
                        _lane_args(run_root=tmp_path), time.monotonic(), [])

    # Check that only the run with raw arrays is skipped
    assert calls == [[remote_job.sys.executable, "scripts/cam_zoom.py", "--run", str(tmp_path / name)]
                     for name in ("old_report", "fresh")]


def test_seed_and_carry_over_never_overwrite(tmp_path):
    # Seed data into a folder that already has a train.txt
    seed, dest = tmp_path / "seed", tmp_path / "drive"
    _write_source(seed, zipped=True)
    dest.mkdir()
    (dest / "train.txt").write_text("kept")
    remote_job.seed_data(seed, dest)
    assert (dest / "train.txt").read_text() == "kept" and (dest / "Train.zip").exists()

    # Carry over runs into a folder that already has run_b
    previous, runs = tmp_path / "prev", tmp_path / "runs"
    (previous / "run_a").mkdir(parents=True)
    (previous / "run_a" / "last.pt").write_text("old")
    (runs / "run_b").mkdir(parents=True)
    remote_job.carry_over_runs(previous, runs)
    assert (runs / "run_a" / "last.pt").read_text() == "old" and (runs / "run_b").is_dir()


def test_carry_over_drops_finished_weights_and_partial_writes(tmp_path):
    # Build two finished runs and one paused run
    previous, runs = tmp_path / "prev", tmp_path / "runs"
    for name in ("done", "zoomed", "paused"):
        (previous / name).mkdir(parents=True)
        (previous / name / "best_ema.pt").write_text("w")
        (previous / name / "summary.json").write_text("{}")
    for name in ("done", "zoomed"):
        (previous / name / "DONE").write_text("")
    (previous / "paused" / "last.pt").write_text("state")
    (previous / "paused" / "last.pt.tmp").write_text("partial")

    # Carry them over, keeping the weights of the zoomed run
    remote_job.carry_over_runs(previous, runs, frozenset({"zoomed"}))
    assert not (runs / "done" / "best_ema.pt").exists() and (runs / "done" / "summary.json").exists()
    assert (runs / "zoomed" / "best_ema.pt").exists()
    assert (runs / "paused" / "last.pt").exists() and (runs / "paused" / "best_ema.pt").exists()
    assert not (runs / "paused" / "last.pt.tmp").exists()
