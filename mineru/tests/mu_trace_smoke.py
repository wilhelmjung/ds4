#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "mineru" / "tests" / "mu_trace.py"
MINERU_PY = Path("/Users/will/github/mineru-model/.venv/bin/python")
MODEL_DIR = Path("/Users/will/github/mineru-model/models")


def require_keys(obj: dict, keys: list[str]) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise AssertionError(f"missing keys: {missing}")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "text_trace.json"
        cmd = [
            str(MINERU_PY),
            str(TRACE),
            "--model-dir",
            str(MODEL_DIR),
            "--mode",
            "text",
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
            "--out",
            str(out),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        data = json.loads(out.read_text(encoding="utf-8"))

    require_keys(
        data,
        [
            "trace_version",
            "mode",
            "model_dir",
            "versions",
            "prompt",
            "chat_prompt",
            "input_ids",
            "attention_mask",
            "position_ids",
            "top_logits",
            "generated_ids",
            "generated_text",
        ],
    )
    assert data["trace_version"] == 1
    assert data["mode"] == "text"
    assert isinstance(data["input_ids"], list) and data["input_ids"]
    assert isinstance(data["generated_ids"], list)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "layout_trace.json"
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        cmd = [
            str(MINERU_PY),
            str(TRACE),
            "--model-dir",
            str(MODEL_DIR),
            "--mode",
            "layout",
            "--device",
            device,
            "--max-new-tokens",
            "4",
            "--out",
            str(out),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        layout = json.loads(out.read_text(encoding="utf-8"))

    require_keys(
        layout,
        [
            "image_grid_thw",
            "pixel_values_shape",
            "vision_patch_embeds_shape",
            "vision_patch_embeds_sample",
            "vision_rotary_pos_emb_shape",
            "vision_rotary_pos_emb_sample",
            "vision_block0_norm1_sample",
            "vision_block0_qkv_token0_sample",
            "vision_block0_attn_output_sample",
            "vision_block0_output_sample",
            "vision_block0_output_token1_sample",
            "vision_block0_tiny_output_file",
            "vision_block0_tiny_output_shape",
            "vision_block0_tiny_output_sample",
            "vision_tiny_last_block_output_file",
            "vision_tiny_last_block_output_shape",
            "vision_tiny_last_block_output_sample",
            "vision_tiny_image_embeds_file",
            "vision_tiny_image_embeds_shape",
            "vision_tiny_image_embeds_sample",
            "vision_last_block_output_sample",
            "image_embeds_shape",
            "image_embeds_sample",
            "layout_generated_ids",
            "layout_generated_text",
        ],
    )
    assert layout["mode"] == "layout"
    assert layout["vision_patch_embeds_shape"] == [5476, 1280]
    assert layout["vision_block0_tiny_output_shape"] == [16, 1280]
    assert layout["vision_tiny_last_block_output_shape"] == [16, 1280]
    assert layout["vision_tiny_image_embeds_shape"] == [4, 896]
    assert layout["image_embeds_shape"] == [1369, 896]
    assert isinstance(layout["layout_generated_ids"], list) and layout["layout_generated_ids"]
    print("mu_trace_smoke ok")


if __name__ == "__main__":
    main()
