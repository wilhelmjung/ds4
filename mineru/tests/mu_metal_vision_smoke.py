#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MU = ROOT / "mu"
TRACE = ROOT / "mineru" / "tests" / "mu-traces" / "layout.json"


def main() -> None:
    env = os.environ.copy()
    env["MU_CHECK_TRACE_SCOPE"] = "vision-block0-norm"
    env["MU_METAL_DEBUG"] = "1"
    result = subprocess.run(
        [
            str(MU),
            "--backend",
            "metal",
            "--no-cpu-fallback",
            "--check-trace",
            str(TRACE),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"metal vision smoke failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert "trace layout vision block0 norm ok" in result.stdout
    assert "mu metal stage: vision_block0_norm1" in result.stderr
    assert "fallback" not in result.stderr
    print("mu_metal_vision_smoke ok")


if __name__ == "__main__":
    main()
