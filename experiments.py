"""Invoke tasks for running experiments."""
from invoke import task


def _maybe_add_device(cmd, device=None):
    if device is not None:
        cmd = cmd.rstrip() + f" --device {device}"
    return cmd


@task
def faithfulness(c, device=None):
    """Run faithfulness experiment."""
    cmd = f"python -m scripts.evaluate -b faithfulness -n faithfulness"
    cmd = _maybe_add_device(cmd, device=device)
    c.run(cmd)


@task
def reconstruction(c, device=None):
    """Run reconstruction experiment."""
    cmd = f"python -m scripts.evaluate -b reconstruction -n reconstruction"
    cmd = _maybe_add_device(cmd, device=device)
    c.run(cmd)


@task
def causality(c, device=None):
    """Run causality experiment."""
    cmd = f"python -m scripts.evaluate -b causality -n causality"
    cmd = _maybe_add_device(cmd, device=device)
    c.run(cmd)
