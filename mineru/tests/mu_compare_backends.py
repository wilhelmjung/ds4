#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MU = ROOT / "mu"
PDF = Path("/Users/will/github/mineru-model/testdata/nasa_systems_engineering_handbook_rev2.pdf")
SAMPLE_IMAGE = Path("/Users/will/github/mineru-model/sample_page.png")


def render_page(page_number: int, out_path: Path) -> None:
    doc = fitz.open(PDF)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
    Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--page", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--skip-content", action="store_true", default=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        image = args.image
        if args.page is not None:
            image = Path(td) / f"page_{args.page:04d}.png"
            render_page(args.page, image)
        if image is None:
            image = SAMPLE_IMAGE

        cmd = [
            str(MU),
            "--compare-backends",
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--image",
            str(image),
        ]
        if args.skip_content:
            cmd.insert(2, "--skip-content")
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"mu compare backends failed with {result.returncode}\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        summary = json.loads(result.stdout)
        assert summary["block_count_equal"] is True
        assert summary["type_equal"] is True
        assert summary["content_equal"] is True
        assert summary["metal_cpu_fallback_count"] == 0
    print("mu_compare_backends ok")


if __name__ == "__main__":
    main()
