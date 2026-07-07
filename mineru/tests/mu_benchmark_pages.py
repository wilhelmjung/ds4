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
PROFILE_RE = re.compile(r"^mu_profile\s+stage=([A-Za-z0-9_.:-]+)\s*(.*)$", re.MULTILINE)


def parse_multipage_stderr(stderr: str) -> dict[str, list[str]]:
    page_stderr_lines = {}
    current_page = None
    for line in stderr.splitlines():
        if line.startswith("mu_page_start page="):
            current_page = line.split("=", 1)[1].strip()
            page_stderr_lines[current_page] = []
        elif line.startswith("mu_page_end page="):
            current_page = None
        elif current_page is not None:
            page_stderr_lines[current_page].append(line)
    return page_stderr_lines


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


def parse_profiles(stderr: str) -> dict[str, list[dict[str, str]]]:
    profiles: dict[str, list[dict[str, str]]] = {}
    for match in PROFILE_RE.finditer(stderr):
        stage = match.group(1)
        fields = {}
        for part in match.group(2).split():
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        profiles.setdefault(stage, []).append(fields)
    return profiles


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
    profiles = parse_profiles(result.stderr)
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
    if profiles:
        row["profiles"] = profiles
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


def run_warmups(cmd: list[str], args: argparse.Namespace, env: dict[str, str] | None) -> list[dict]:
    rows = []
    for run_idx in range(1, args.warmup_runs + 1):
        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
                env=env,
            )
            returncode = result.returncode
            stderr_out = result.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = -1
            stderr_out = exc.stderr or ""
            if isinstance(stderr_out, bytes):
                stderr_out = stderr_out.decode("utf-8", errors="replace")
        row = {
            "run": run_idx,
            "returncode": returncode,
            "run_wall_seconds": time.perf_counter() - start,
            "stderr_tail": stderr_out[-4000:],
        }
        rows.append(row)
        print(json.dumps({"warmup": row}, ensure_ascii=False), flush=True)
        if returncode != 0 and not args.keep_going:
            raise SystemExit(f"warmup run {run_idx} failed with code {returncode}")
    return rows


def summarize(args: argparse.Namespace, rows: list[dict],
              warmup_rows: list[dict] | None = None) -> dict:
    completed = [r for r in rows if r.get("returncode") == 0 and "error" not in r]
    total = sum(float(r["seconds"]) for r in completed)
    mean = total / len(completed) if completed else 0.0
    run_wall_seconds = [
        float(r["run_wall_seconds"])
        for r in rows
        if "run_wall_seconds" in r
    ]
    timing_sums: dict[str, float] = {}
    timing_counts: dict[str, int] = {}
    for row in completed:
        for stage, seconds in dict(row.get("stage_timings") or {}).items():
            timing_sums[stage] = timing_sums.get(stage, 0.0) + float(seconds)
            timing_counts[stage] = timing_counts.get(stage, 0) + 1
    summary = {
        "backend": args.backend,
        "pages": args.pages,
        "max_new_tokens": args.max_new_tokens,
        "content_max_new_tokens": args.content_max_new_tokens,
        "skip_content": args.skip_content,
        "timing": args.timing,
        "warmup_runs": args.warmup_runs,
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
    if run_wall_seconds:
        summary["run_wall_seconds"] = max(run_wall_seconds)
    if warmup_rows:
        summary["warmup_rows"] = warmup_rows
    return summary


def write_summary(args: argparse.Namespace, rows: list[dict],
                  warmup_rows: list[dict] | None = None) -> None:
    Path(args.out).write_text(
        json.dumps(summarize(args, rows, warmup_rows), indent=2, ensure_ascii=False),
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
    parser.add_argument("--threads", type=int, default=1)
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
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--kv-cache-bf16", action="store_true")
    parser.add_argument("--use-icb", action="store_true")
    args = parser.parse_args()
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")

    rows = load_resume_rows(args)
    done_pages = {
        int(r["page"])
        for r in rows
        if r.get("returncode") == 0 and "error" not in r and "page" in r
    }

    pages_to_run = [p for p in args.pages if p not in done_pages]
    if not pages_to_run:
        print("All pages already processed.")
        write_summary(args, rows)
        return

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        images_to_run = []
        for page in pages_to_run:
            image = tmp / f"{args.backend}_page_{page:04d}.png"
            render_page(page, image)
            images_to_run.append(image)

        # Build CLI command
        cmd = [str(MU), "--backend", args.backend]
        if args.threads is not None:
            cmd.extend(["--threads", str(args.threads)])
        if args.backend == "metal":
            cmd.append("--no-cpu-fallback")
        if args.max_new_tokens is not None:
            cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
        if args.skip_content:
            cmd.append("--skip-content")
        if args.markdown:
            cmd.append("--markdown")
        else:
            cmd.append("--json")
        if args.kv_cache_bf16:
            cmd.append("--kv-cache-bf16")
        if args.use_icb:
            cmd.append("--use-icb")

        for img in images_to_run:
            cmd.extend(["--image", str(img)])

        if args.save_output_dir is not None:
            out_dir = args.save_output_dir
        else:
            out_dir = tmp / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output-dir", str(out_dir)])

        env = None
        if args.content_max_new_tokens is not None or args.timing:
            env = os.environ.copy()
        if args.content_max_new_tokens is not None:
            env["MU_CONTENT_MAX_NEW_TOKENS"] = str(args.content_max_new_tokens)
        if args.timing:
            env["MU_TIMING"] = "1"

        warmup_rows = run_warmups(cmd, args, env) if args.warmup_runs else []

        print(f"Running command: {' '.join(cmd)}")
        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
                env=env,
            )
            returncode = result.returncode
            stderr_out = result.stderr
            run_wall_seconds = time.perf_counter() - start
        except subprocess.TimeoutExpired as exc:
            returncode = -1
            run_wall_seconds = time.perf_counter() - start
            stderr_out = exc.stderr or ""
            if isinstance(stderr_out, bytes):
                stderr_out = stderr_out.decode("utf-8", errors="replace")
            print(f"Process timed out: {exc}")

        # Parse stderr by page
        page_stderr_lines = parse_multipage_stderr(stderr_out)

        # Process each page row
        for idx, page in enumerate(pages_to_run):
            img_path_str = str(images_to_run[idx])
            page_lines = page_stderr_lines.get(img_path_str, [])
            page_stderr = "\n".join(page_lines)
            stage_timings = parse_stage_timings(page_stderr)
            profiles = parse_profiles(page_stderr)
            fallback_detected = "fallback" in page_stderr.lower()

            row = {
                "page": page,
                "command": cmd,
                "returncode": returncode,
                "content_max_new_tokens": args.content_max_new_tokens,
                "fallback_detected": fallback_detected,
                "run_wall_seconds": run_wall_seconds,
                "stderr_tail": page_stderr[-4000:],
            }

            if stage_timings:
                row["stage_timings"] = stage_timings
                row["seconds"] = stage_timings.get("page_total", 0.0)
            else:
                row["seconds"] = 0.0
            if profiles:
                row["profiles"] = profiles

            if returncode != 0:
                row["error"] = f"mu process failed/timed out with code {returncode}"
            elif img_path_str not in page_stderr_lines:
                row["error"] = "page execution did not start or complete"
            elif args.backend == "metal" and fallback_detected:
                row["error"] = "metal fallback appeared in page stderr"
            else:
                # Read outputs from generated file
                out_ext = "md" if args.markdown else "json"
                out_filename = out_dir / f"{images_to_run[idx].stem}.{out_ext}"
                if out_filename.exists():
                    if not args.markdown:
                        try:
                            blocks = json.loads(out_filename.read_text(encoding="utf-8"))
                            row["blocks"] = len(blocks)
                            row["types"] = [b.get("type") for b in blocks]
                            row["output_json"] = str(out_filename)
                        except Exception as e:
                            row["error"] = f"failed to parse json output: {e}"
                    else:
                        row["blocks"] = 0
                        row["types"] = []
                        row["output_json"] = str(out_filename)
                else:
                    row["error"] = f"output file {out_filename} was not created"

            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            write_summary(args, rows, warmup_rows)
            if "error" in row and not args.keep_going:
                raise SystemExit(row["error"])

    write_summary(args, rows, warmup_rows)


if __name__ == "__main__":
    main()
