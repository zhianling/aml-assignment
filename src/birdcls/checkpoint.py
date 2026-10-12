"""Crash-safe checkpointing: atomic saves, an asynchronous writer, RNG state and fp16 weights."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    """Capture the Python, NumPy, torch and (if available) CUDA random states."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the random states captured by ``rng_state``."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(obj: Any, path: Path) -> None:
    """Save ``obj`` to ``path`` through a temporary file, so an interrupted write keeps the old file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class AsyncCheckpointWriter:
    """Save to fast local disk, then copy to the slow destination (Drive) in a background thread.

    Training does not stall on a multi-GB upload, and each copy goes through ``<name>.tmp`` and an
    atomic rename, so a disconnect mid-upload never destroys the last good checkpoint.
    """

    def __init__(self) -> None:
        """Create the local scratch directory the saves are serialized into."""
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._scratch = Path(tempfile.mkdtemp(prefix="ckpt_"))

    def save(self, obj: Any, path: Path) -> None:
        """Serialize ``obj`` locally, then upload it to ``path`` in the background."""
        self.wait()  # at most one upload in flight; also surfaces errors from the previous one
        path = Path(path)
        local = self._scratch / path.name
        torch.save(obj, local)

        def upload() -> None:
            """Copy the local file to ``path`` atomically and record any error for ``wait``."""
            try:
                tmp = path.with_name(path.name + ".tmp")
                shutil.copyfile(local, tmp)
                os.replace(tmp, path)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the training thread
                self._error = exc
            finally:
                local.unlink(missing_ok=True)

        self._thread = threading.Thread(target=upload, name=f"upload-{path.name}", daemon=False)
        self._thread.start()

    def wait(self) -> None:
        """Block until the running upload finishes and raise if it failed."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("Background checkpoint upload failed") from error

    def close(self) -> None:
        """Finish the pending upload and delete the scratch directory."""
        self.wait()
        shutil.rmtree(self._scratch, ignore_errors=True)


def half_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a weights-only fp16 CPU copy of ``module``, about 1/4 the size of a full training checkpoint."""
    return {k: (v.half() if v.is_floating_point() else v).cpu() for k, v in module.state_dict().items()}
