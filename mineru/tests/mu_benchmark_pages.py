#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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
TIMING_RE = re.compile(
    r"^mu_timing\s+stage=([A-Za-z0-9_.:-]+)\s+seconds=([0-9]+(?:\.[0-9]+)?)$",
    re.MULTILINE,
)


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


def parse_stage_timings(stderr: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for match in TIMING_RE.finditer(stderr):
        stage = match.group(1)
        timings[stage] = timings.get(stage, 0.0) + float(match.group(2))
    return timings


def run_one(
    backend: str,
    image: Path,
    *,
    max_new_tokens: int | None,
    skip_content: bool,
    timeout: int,
    output_dir: Path | None = None,
    page: int | None = None,
    content_max_new_tokens: int | None = None,
    timing: bool = False,
) -> dict:
    cmd = [str(MU), "--backend", backend]
    if backend == "metal":
        cmd.append("--no-cpu-fallback")
    if max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])
    if skip_content:
        cmd.append("--skip-content")
    cmd.extend(["--image", str(image), "--json"])

    start = time.perf_counter()
    env = None
    if content_max_new_tokens is not None or timing:
        env = os.environ.copy()
    if content_max_new_tokens is not None:
        env["MU_CONTENT_MAX_NEW_TOKENS"] = str(content_max_new_tokens)
    if timing:
        env["MU_TIMING"] = "1"
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "seconds": elapsed,
            "command": cmd,
            "returncode": None,
            "timeout_seconds": timeout,
            "content_max_new_tokens": content_max_new_tokens,
            "fallback_detected": "fallback" in stderr.lower(),
            "stderr_tail": stderr[-4000:],
            "stdout_tail": stdout[-4000:],
            "error": f"mu timed out after {timeout}s",
        }
    elapsed = time.perf_counter() - start
    fallback_detected = "fallback" in result.stderr.lower()
    stage_timings = parse_stage_timings(result.stderr)
    row = {
        "seconds": elapsed,
        "command": cmd,
        "returncode": result.returncode,
        "content_max_new_tokens": content_max_new_tokens,
        "fallback_detected": fallback_detected,
        "stderr_tail": result.stderr[-4000:],
    }
    if stage_timings:
        row["stage_timings"] = stage_timings
    if result.returncode != 0:
        row["error"] = f"mu exited with {result.returncode}"
        row["stdout_tail"] = result.stdout[-4000:]
        return row
    if backend == "metal" and fallback_detected:
        row["error"] = "metal fallback appeared in stderr"
        row["stdout_tail"] = result.stdout[-4000:]
        return row
    blocks = json.loads(result.stdout)
    row["blocks"] = len(blocks)
    row["types"] = [b.get("type") for b in blocks]
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{backend}_page_{page:04d}" if page is not None else f"{backend}_output"
        out_path = output_dir / f"{stem}.json"
        out_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        row["output_json"] = str(out_path)
    return row


def summarize(args: argparse.Namespace, rows: list[dict]) -> dict:
    completed = [r for r in rows if r.get("returncode") == 0 and "error" not in r]
    total = sum(float(r["seconds"]) for r in completed)
    mean = total / len(completed) if completed else 0.0
    timing_sums: dict[str, float] = {}
    timing_counts: dict[str, int] = {}
    for row in completed:
        for stage, seconds in dict(row.get("stage_timings") or {}).items():
            timing_sums[stage] = timing_sums.get(stage, 0.0) + float(seconds)
            timing_counts[stage] = timing_counts.get(stage, 0) + 1
    return {
        "backend": args.backend,
        "pages": args.pages,
        "max_new_tokens": args.max_new_tokens,
        "content_max_new_tokens": args.content_max_new_tokens,
        "skip_content": args.skip_content,
        "timing": args.timing,
        "total_seconds": total,
        "mean_seconds": mean,
        "mean_stage_timings": {
            stage: timing_sums[stage] / timing_counts[stage]
            for stage in sorted(timing_sums)
        },
        "completed_pages": len(completed),
        "failed_pages": len(rows) - len(completed),
        "fallback_rows": sum(1 for r in rows if r.get("fallback_detected")),
        "rows": rows,
    }


def write_summary(args: argparse.Namespace, rows: list[dict]) -> None:
    Path(args.out).write_text(
        json.dumps(summarize(args, rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_resume_rows(args: argparse.Namespace) -> list[dict]:
    path = Path(args.out)
    if not args.resume or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("backend") != args.backend:
        raise SystemExit(f"cannot resume {path}: backend mismatch")
    if data.get("max_new_tokens") != args.max_new_tokens:
        raise SystemExit(f"cannot resume {path}: max_new_tokens mismatch")
    if data.get("content_max_new_tokens") != args.content_max_new_tokens:
        raise SystemExit(f"cannot resume {path}: content_max_new_tokens mismatch")
    if data.get("skip_content") != args.skip_content:
        raise SystemExit(f"cannot resume {path}: skip_content mismatch")
    if data.get("timing") != args.timing:
        raise SystemExit(f"cannot resume {path}: timing mismatch")
    return list(data.get("rows", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pages", type=parse_pages,
                        default=DEFAULT_PAGES)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--content-max-new-tokens", type=int)
    parser.add_argument("--skip-content", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--save-output-dir", type=Path)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()

    rows = load_resume_rows(args)
    done_pages = {
        int(r["page"])
        for r in rows
        if r.get("returncode") == 0 and "error" not in r and "page" in r
    }
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for page in args.pages:
            if page in done_pages:
                continue
            image = tmp / f"page_{page:04d}.png"
            render_page(page, image)
            row = run_one(
                args.backend,
                image,
                max_new_tokens=args.max_new_tokens,
                skip_content=args.skip_content,
                timeout=args.timeout,
                output_dir=args.save_output_dir,
                page=page,
                content_max_new_tokens=args.content_max_new_tokens,
                timing=args.timing,
            )
            row["page"] = page
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            write_summary(args, rows)
            if "error" in row and not args.keep_going:
                raise SystemExit(row["error"])
    write_summary(args, rows)


if __name__ == "__main__":
    main()
