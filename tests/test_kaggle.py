import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from birdcls.config import REPO_ROOT
from script_loader import load_script

kernel, push = load_script("kaggle_kernel"), load_script("kaggle_push")


def _data(root):
    """Create the dataset folder the way Kaggle mounts it under ``root``."""
    # Kaggle unpacks an uploaded Train.zip into Train/Train/.
    (root / "cos30082-cub200-2010" / "Train" / "Train").mkdir(parents=True)
    (root / "cos30082-cub200-2010" / "train.txt").write_text("a.jpg 0\n")
    (root / "cos30082-cub200-2010" / "test.txt").write_text("b.jpg 0\n")


@pytest.mark.parametrize("zipped", [True, False])
def test_locate_inputs_finds_code_data_and_previous_runs(tmp_path, zipped):
    # Build the data, the code (zipped or unpacked) and one previous run
    _data(tmp_path)
    code_root = tmp_path / "cos30082-bird-code"
    code_root.mkdir()
    if zipped:
        with zipfile.ZipFile(code_root / "code.zip", "w") as zf:
            zf.writestr("kaggle_job.json", "{}")
            zf.writestr("src/birdcls/__init__.py", "")
    else:
        (code_root / "code").mkdir()
        (code_root / "code" / "kaggle_job.json").write_text("{}")
    (tmp_path / "cos30082-bird-training" / "runs" / "run_a").mkdir(parents=True)
    (tmp_path / "cos30082-bird-training" / "runs" / "run_a" / "config.json").write_text("{}")

    # Locate the inputs and stage the code
    code, data, previous = kernel.locate_inputs(tmp_path)
    assert data == tmp_path / "cos30082-cub200-2010"
    assert previous == tmp_path / "cos30082-bird-training" / "runs"
    kernel.stage_code(code, tmp_path / "staged")
    assert (tmp_path / "staged" / "kaggle_job.json").exists()


def test_first_session_has_no_previous_runs(tmp_path):
    _data(tmp_path)
    assert kernel.locate_inputs(tmp_path)[2] is None


def test_pip_requirements_keep_kaggles_torch():
    reqs = push.pip_requirements(REPO_ROOT / "pyproject.toml")
    assert any(r.startswith("timm") for r in reqs)
    assert not any(r.startswith(("torch>", "torch=", "torchvision")) for r in reqs)


def test_kernel_metadata_attaches_its_own_output_only_when_resuming():
    first = push.kernel_metadata("me", resume=False, machine_shape="NvidiaTeslaT4")
    again = push.kernel_metadata("me", resume=True, machine_shape="NvidiaTeslaT4")
    assert first["kernel_sources"] == [] and again["kernel_sources"] == [first["id"]]
    assert first["is_private"] and first["enable_gpu"] and first["dataset_sources"][0].startswith("me/")


def test_code_zip_holds_the_repository_and_the_job(tmp_path):
    job = push.job_spec(["configs/resnet50_448.toml"], "abc1234", 11.0, [], zoom=["eva02_l_inat21_336"])
    push.build_code_zip(tmp_path / "code.zip", job)
    with zipfile.ZipFile(tmp_path / "code.zip") as zf:
        packed = json.loads(zf.read("kaggle_job.json"))
        assert packed["configs"] == ["configs/resnet50_448.toml"] and packed["zoom"] == ["eva02_l_inat21_336"]
        assert "src/birdcls/train.py" in zf.namelist()


def test_code_upload_waits_until_kaggle_serves_the_new_job(tmp_path, monkeypatch):
    # Right after `datasets version`, status already says "ready" but downloads still serve the old job.
    old, new = {"commit": "old"}, {"commit": "new"}
    served = [old, old, new]
    calls = []

    def fake_kaggle(*args, check):
        """Stand in for the kaggle CLI: each download serves the next job in ``served``."""
        calls.append(args[:2])
        if args[:2] == ("datasets", "download"):
            dest = args[args.index("-p") + 1]
            (Path(dest) / "kaggle_job.json").write_text(json.dumps(served.pop(0)))
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")
        return subprocess.CompletedProcess([], 0, stdout='status "ready"', stderr="")

    # Upload and wait for the new job
    monkeypatch.setattr(push, "kaggle", fake_kaggle)
    monkeypatch.setattr(push.time, "sleep", lambda s: None)
    push.upload_dataset(tmp_path, "me/code", "code", "commit new", job=new)
    assert served == [] and calls.count(("datasets", "download")) == 3


def test_status_retries_network_errors_and_only_403_means_missing(monkeypatch):
    # Script the replies of each reference
    replies = {
        "flaky": [(1, "SSLError: EOF occurred in violation of protocol"), (0, 'has status "RUNNING"')],
        "missing": [(1, "403 Client Error: Forbidden")],
        "down": [(1, "Max retries exceeded")] * 4,
    }

    def fake_kaggle(kind, _, ref, check):
        """Stand in for the kaggle CLI: return the next scripted (exit code, text) reply for ``ref``."""
        code, text = replies[ref].pop(0)
        return subprocess.CompletedProcess([], code, stdout=text, stderr="")

    monkeypatch.setattr(push, "kaggle", fake_kaggle)
    monkeypatch.setattr(push.time, "sleep", lambda s: None)

    # Check each outcome
    assert "running" in push.status("kernels", "flaky")
    assert push.status("datasets", "missing") is None
    with pytest.raises(SystemExit, match="kept failing"):
        push.status("kernels", "down")
