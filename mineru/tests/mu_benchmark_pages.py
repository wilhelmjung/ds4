#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MU = ROOT / "mu"
PDF = Path("/Users/will/github/mineru-model/testdata/nasa_systems_engineering_handbook_rev2.pdf")
DEFAULT_PAGES = [224, 234, 237, 241, 244, 247, 258, 281, 303, 334]


def parse_pages(text: str) -> list[int]:
    pages: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        pages.append(int(part))
    if not pages:
        raise argparse.ArgumentTypeError("at least one page is required")
    return pages


def render_page(page_number: int, out_path: Path) -> None:
    doc = fitz.open(PDF)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
    Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(out_path)


def run_one(
    backend: str,
    image: Path,
    *,
    max_new_tokens: int | None,
    skip_content: bool,
    timeout: int,
) -> tuple[float, list[dict], bool]:
    cmd = [str(MU), "--backend", backend]
    if backend == "metal":
        cmd.append("--no-cpu-fallback")
    if max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])
    if skip_content:
        cmd.append("--skip-content")
    cmd.extend(["--image", str(image), "--json"])

    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    fallback_detected = "fallback" in result.stderr.lower()
    if backend == "metal" and fallback_detected:
        raise RuntimeError(f"metal fallback appeared in stderr:\n{result.stderr}")
    return elapsed, json.loads(result.stdout), fallback_detected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pages", type=parse_pages,
                        default=DEFAULT_PAGES)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--skip-content", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for page in args.pages:
            image = tmp / f"page_{page:04d}.png"
            render_page(page, image)
            seconds, blocks, fallback_detected = run_one(
                args.backend,
                image,
                max_new_tokens=args.max_new_tokens,
                skip_content=args.skip_content,
                timeout=args.timeout,
            )
            row = {
                "page": page,
                "seconds": seconds,
                "blocks": len(blocks),
                "types": [b.get("type") for b in blocks],
                "fallback_detected": fallback_detected,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    total = sum(r["seconds"] for r in rows)
    output = {
        "backend": args.backend,
        "pages": args.pages,
        "max_new_tokens": args.max_new_tokens,
        "skip_content": args.skip_content,
        "total_seconds": total,
        "mean_seconds": total / len(rows),
        "fallback_rows": sum(1 for r in rows if r["fallback_detected"]),
        "rows": rows,
    }
    Path(args.out).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
