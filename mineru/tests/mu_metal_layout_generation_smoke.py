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
    env["MU_CHECK_TRACE_SCOPE"] = "layout-generation"
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
        timeout=600,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"metal layout generation smoke failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert "trace layout logits ok" in result.stdout
    assert "trace layout generation ok" in result.stdout
    assert "mu metal stage: text_logits" in result.stderr
    assert "mu metal stage: text_cached_attn" in result.stderr
    assert "mu metal stage: text_generate" in result.stderr
    assert "fallback" not in result.stderr
    print("mu_metal_layout_generation_smoke ok")


if __name__ == "__main__":
    main()
