"""Device and mixed-precision selection shared by every entry point (Mac, Colab, CPU tests)."""

from __future__ import annotations

import torch


def pick_device(name: str = "auto") -> torch.device:
    """Return the device called ``name``, or with ``auto`` the best available of CUDA, MPS and CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait until all work queued on ``device`` has finished.

    CUDA and MPS run asynchronously, so a timer read without this measures only the queuing.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_amp(setting: str, device: torch.device) -> torch.dtype | None:
    """Return the autocast dtype for ``setting`` on ``device`` (None = full fp32).

    ``auto`` picks bf16 on Ampere or newer GPUs and fp16 on older ones such as Colab's T4,
    which has fp16 tensor cores but no native bf16. fp16 has a narrow exponent range, so it
    must be paired with a GradScaler; bf16 does not need one. On MPS and CPU, ``auto`` stays
    in fp32 because mixed-precision training there is slower or less mature than on CUDA.
    """
    if setting == "none":
        return None
    if setting == "fp16":
        return torch.float16
    if setting == "bf16":
        return torch.bfloat16
    if setting != "auto":
        raise ValueError(f"Unknown amp setting {setting!r}")
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return None
