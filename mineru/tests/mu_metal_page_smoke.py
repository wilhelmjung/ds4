#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MU = ROOT / "mu"
IMAGE = Path("/Users/will/github/mineru-model/sample_page.png")


def main() -> None:
    result = subprocess.run(
        [
            str(MU),
            "--backend",
            "metal",
            "--no-cpu-fallback",
            "--max-new-tokens",
            "4",
            "--skip-content",
            "--image",
            str(IMAGE),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1800,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"metal page smoke failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    blocks = json.loads(result.stdout)
    assert isinstance(blocks, list)
    assert "fallback" not in result.stderr
    print("mu_metal_page_smoke ok")


if __name__ == "__main__":
    main()
