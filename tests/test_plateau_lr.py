import pytest
import torch

from birdcls.config import TrainConfig
from birdcls.train import PlateauLR


def _optimizer():
    """Build an AdamW optimizer with a plain group and a half-scaled group."""
    params = [torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))]
    # The second group mimics a timm layer-decay group, whose LR multiplier is "lr_scale".
    return torch.optim.AdamW([{"params": params[:1]}, {"params": params[1:], "lr_scale": 0.5}], lr=1e-3)


def _lrs(opt):
    """Return the current LR of each parameter group."""
    return [g["lr"] for g in opt.param_groups]


def test_warm_up_then_cut_at_each_stall_down_to_min_lr():
    # Warm up from min_lr to the full LR
    opt = _optimizer()
    sched = PlateauLR(opt, min_lr=1e-6, warmup_updates=10, factor=0.1, patience=2)
    assert _lrs(opt) == pytest.approx([1e-6, 1e-6])
    sched.step_update(5)
    assert _lrs(opt) == pytest.approx([1e-6 + (1e-3 - 1e-6) / 2, 1e-6 + (5e-4 - 1e-6) / 2])
    sched.step_update(20)
    assert _lrs(opt) == pytest.approx([1e-3, 5e-4])

    # Cut the LR after each run of stalls
    assert [sched.step_epoch(g) for g in (True, False, True, False)] == [False] * 4  # a gain resets the count
    assert sched.step_epoch(False) and _lrs(opt) == pytest.approx([1e-4, 5e-5])  # applied before the next update
    assert [sched.step_epoch(False) for _ in range(2)] == [False, True]

    # Check the LR stops at min_lr
    for _ in range(6):
        sched.step_epoch(False)
    assert _lrs(opt) == pytest.approx([1e-6, 1e-6])


def test_state_round_trip_continues_the_same_schedule():
    # Train a schedule through one cut and one stall
    sched = PlateauLR(_optimizer(), min_lr=0.0, warmup_updates=0, factor=0.1, patience=2)
    for _ in range(3):
        sched.step_epoch(False)  # one cut, then one stall toward the next

    # Resume a new schedule from its state
    opt = _optimizer()
    resumed = PlateauLR(opt, min_lr=0.0, warmup_updates=0, factor=0.1, patience=2)
    resumed.load_state_dict(sched.state_dict())
    resumed.step_update(100)

    # Check the cut LR and the pending stall carried over
    assert _lrs(opt) == pytest.approx([1e-4, 5e-5])
    assert resumed.step_epoch(False)


def test_plateau_schedule_needs_a_validation_split():
    with pytest.raises(ValueError, match="use_val"):
        TrainConfig(run_name="x", backbone="resnet50", lr_schedule="plateau", use_val=False)
    with pytest.raises(ValueError, match="lr_schedule"):
        TrainConfig(run_name="x", backbone="resnet50", lr_schedule="step")
