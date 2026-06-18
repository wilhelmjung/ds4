#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MU = ROOT / "mu"
TRACE = ROOT / "mineru" / "tests" / "mu-traces" / "text.json"


def main() -> None:
    env = os.environ.copy()
    env["MU_CHECK_TRACE_SCOPE"] = "text-layer0-seq"
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
            f"metal text smoke failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert "trace text layer0 qkv ok" in result.stdout
    assert "trace text layer0 attn ok" in result.stdout
    assert "trace text layer0 mlp ok" in result.stdout
    assert "trace text layer0 seq ok" in result.stdout
    assert "mu metal stage: text_layer0_input_norm" in result.stderr
    assert "mu metal stage: text_layer0_q_proj" in result.stderr
    assert "mu metal stage: text_layer0_k_proj" in result.stderr
    assert "mu metal stage: text_layer0_v_proj" in result.stderr
    assert "mu metal stage: text_layer0_attn_token0" in result.stderr
    assert "mu metal stage: text_layer0_o_proj" in result.stderr
    assert "mu metal stage: text_layer0_attn_residual" in result.stderr
    assert "mu metal stage: text_layer0_post_norm" in result.stderr
    assert "mu metal stage: text_layer0_gate_proj" in result.stderr
    assert "mu metal stage: text_layer0_up_proj" in result.stderr
    assert "mu metal stage: text_layer0_silu_mul" in result.stderr
    assert "mu metal stage: text_layer0_down_proj" in result.stderr
    assert "mu metal stage: text_layer0_mlp_residual" in result.stderr
    assert "mu metal stage: text_layer0_seq_input_norm" in result.stderr
    assert "mu metal stage: text_layer0_seq_q_proj" in result.stderr
    assert "mu metal stage: text_layer0_seq_k_proj" in result.stderr
    assert "mu metal stage: text_layer0_seq_v_proj" in result.stderr
    assert "mu metal stage: text_layer0_seq_attn" in result.stderr
    assert "mu metal stage: text_layer0_seq_o_proj" in result.stderr
    assert "mu metal stage: text_layer0_seq_attn_residual" in result.stderr
    assert "mu metal stage: text_layer0_seq_post_norm" in result.stderr
    assert "mu metal stage: text_layer0_seq_gate_proj" in result.stderr
    assert "mu metal stage: text_layer0_seq_up_proj" in result.stderr
    assert "mu metal stage: text_layer0_seq_silu_mul" in result.stderr
    assert "mu metal stage: text_layer0_seq_down_proj" in result.stderr
    assert "mu metal stage: text_layer0_seq_mlp_residual" in result.stderr
    assert "fallback" not in result.stderr
    print("mu_metal_text_smoke ok")


if __name__ == "__main__":
    main()
