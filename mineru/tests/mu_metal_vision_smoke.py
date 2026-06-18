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
    env["MU_CHECK_TRACE_SCOPE"] = "vision-block0-tiny-output"
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
    assert "trace layout vision block0 qkv ok" in result.stdout
    assert "trace layout vision block0 attn ok" in result.stdout
    assert "trace layout vision block0 tiny output ok" in result.stdout
    assert "mu metal stage: vision_block0_norm1" in result.stderr
    assert "mu metal stage: vision_block0_qkv" in result.stderr
    assert "mu metal stage: vision_block0_attn_norm1" in result.stderr
    assert "mu metal stage: vision_block0_attn_q" in result.stderr
    assert "mu metal stage: vision_block0_attn_kv" in result.stderr
    assert "mu metal stage: vision_block0_attn_concat" in result.stderr
    assert "mu metal stage: vision_block0_attn_proj" in result.stderr
    assert "mu metal stage: vision_block0_output_residual1" in result.stderr
    assert "mu metal stage: vision_block0_output_norm2" in result.stderr
    assert "mu metal stage: vision_block0_output_fc1" in result.stderr
    assert "mu metal stage: vision_block0_output_quick_gelu" in result.stderr
    assert "mu metal stage: vision_block0_output_fc2" in result.stderr
    assert "mu metal stage: vision_block0_output_residual2" in result.stderr
    assert "mu metal stage: vision_block0_output_all_rows" in result.stderr
    assert "fallback" not in result.stderr
    print("mu_metal_vision_smoke ok")


if __name__ == "__main__":
    main()
