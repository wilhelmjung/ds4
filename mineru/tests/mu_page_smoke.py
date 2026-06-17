#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MINERU_ROOT = Path("/Users/will/github/mineru-model")
PDF = MINERU_ROOT / "testdata/nasa_systems_engineering_handbook_rev2.pdf"


def render_page_224(out_path: Path) -> tuple[int, int]:
    doc = fitz.open(PDF)
    page = doc.load_page(223)
    pix = page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
    Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(out_path)
    return pix.width, pix.height


def run_mu(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(ROOT / "mu"), *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"mu {' '.join(args)} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def assert_block_schema(block: dict) -> None:
    assert isinstance(block.get("type"), str) and block["type"]
    bbox = block.get("bbox")
    assert isinstance(bbox, list) and len(bbox) == 4
    x1, y1, x2, y2 = bbox
    assert 0.0 <= x1 < x2 <= 1.0
    assert 0.0 <= y1 < y2 <= 1.0
    assert "content" in block
    if block.get("merge_prev"):
        assert block["type"] == "text"


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        image = Path(td) / "nasa-page-0224.png"
        width, height = render_page_224(image)
        assert (width, height) == (1020, 1320)

        json_out = run_mu("--image", str(image), "--json").stdout
        blocks = json.loads(json_out)
        assert isinstance(blocks, list)
        assert [b["type"] for b in blocks] == ["table", "footer", "page_number"]
        for block in blocks:
            assert_block_schema(block)

        table = blocks[0].get("content") or ""
        footer = blocks[1].get("content") or ""
        assert "<table>" in table
        assert "Cost-Effectiveness Analysis" in table
        assert "Critical Design Review" in table
        assert "NASA Systems Engineering Handbook" in footer

        markdown = run_mu("--image", str(image), "--markdown").stdout
        assert "<table>" in markdown
        assert "Decision Authority" in markdown
        assert "NASA Systems Engineering Handbook" in markdown

    print("mu_page_smoke ok")


if __name__ == "__main__":
    main()
