#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "mineru" / "tests" / "mu-traces"


def run_mu_check(trace: str) -> str:
    result = subprocess.run(
        [str(ROOT / "mu"), "--check-trace", str(TRACE_DIR / trace)],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def assert_metal_layout_trace_if_available() -> None:
    result = subprocess.run(
        [
            str(ROOT / "mu"),
            "--backend",
            "metal",
            "--check-trace",
            str(TRACE_DIR / "layout.json"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if result.returncode != 0:
        if "Metal" in result.stderr or "metal" in result.stderr:
            return
        raise AssertionError(
            "metal layout trace failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert "trace layout logits ok" in result.stdout


def assert_layout_token1_is_checked() -> None:
    trace_path = TRACE_DIR / "layout.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["vision_block0_output_token1_sample"][0] += 10.0
    with tempfile.TemporaryDirectory() as td:
        bad_trace = Path(td) / "layout-token1-bad.json"
        bad_trace.write_text(json.dumps(trace), encoding="utf-8")
        result = subprocess.run(
            [str(ROOT / "mu"), "--check-trace", str(bad_trace)],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    assert result.returncode != 0, "mu accepted a bad block0 token1 trace sample"
    assert "token1" in result.stderr


def assert_image_sidecar_path_runs() -> None:
    env = os.environ.copy()
    env["MU_IMAGE_EMBEDS_FILE"] = str(TRACE_DIR / "layout.image_embeds.f32.bin")
    env["MU_MAX_NEW_TOKENS"] = "4"
    env["MU_SKIP_CONTENT"] = "1"
    result = subprocess.run(
        [
            str(ROOT / "mu"),
            "--image",
            "/Users/will/github/mineru-model/sample_page.png",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    blocks = json.loads(result.stdout)
    assert isinstance(blocks, list)


def assert_sidecar_layout_generates_blocks_fast() -> None:
    env = os.environ.copy()
    env["MU_IMAGE_EMBEDS_FILE"] = str(TRACE_DIR / "layout.image_embeds.f32.bin")
    env["MU_MAX_NEW_TOKENS"] = "64"
    env["MU_SKIP_CONTENT"] = "1"
    result = subprocess.run(
        [
            str(ROOT / "mu"),
            "--image",
            "/Users/will/github/mineru-model/sample_page.png",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=120,
    )
    blocks = json.loads(result.stdout)
    assert isinstance(blocks, list) and blocks


def assert_image_native_path_starts() -> None:
    env = os.environ.copy()
    env.pop("MU_IMAGE_EMBEDS_FILE", None)
    env["MU_MAX_NEW_TOKENS"] = "64"
    env["MU_SKIP_CONTENT"] = "1"
    result = subprocess.run(
        [
            str(ROOT / "mu"),
            "--image",
            "/Users/will/github/mineru-model/sample_page.png",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=180,
    )
    blocks = json.loads(result.stdout)
    assert isinstance(blocks, list) and blocks


def main() -> None:
    text_out = run_mu_check("text.json")
    assert "trace text chat ok" in text_out
    assert "trace text tokenizer ok" in text_out
    assert "trace text positions ok" in text_out
    assert "trace text logits ok" in text_out
    assert "trace text generation ok" in text_out

    layout_out = run_mu_check("layout.json")
    assert "trace layout chat ok" in layout_out
    assert "trace layout tokenizer ok" in layout_out
    assert "trace layout processor ok" in layout_out
    assert "trace layout positions ok" in layout_out
    assert "trace layout vision patch ok" in layout_out
    assert "trace layout vision rope ok" in layout_out
    assert "trace layout vision block0 norm ok" in layout_out
    assert "trace layout vision block0 qkv ok" in layout_out
    assert "trace layout vision block0 attn ok" in layout_out
    assert "trace layout vision block0 output ok" in layout_out
    assert "trace layout generation ok" in layout_out
    assert "trace layout logits ok" in layout_out
    assert "trace layout parser ok" in layout_out
    assert_layout_token1_is_checked()
    assert_image_sidecar_path_runs()
    assert_sidecar_layout_generates_blocks_fast()
    assert_image_native_path_starts()
    assert_metal_layout_trace_if_available()

    print("mu_cli_smoke ok")


if __name__ == "__main__":
    main()
