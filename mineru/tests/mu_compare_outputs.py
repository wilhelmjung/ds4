#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean, median
from typing import Any


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def text_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return [m.group(0).lower() for m in TOKEN_RE.finditer(html.unescape(text))]


def token_f1(ref: str | None, pred: str | None) -> float:
    ref_tokens = text_tokens(ref)
    pred_tokens = text_tokens(pred)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = sum((ref_counts & pred_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


class _CellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_cell = 0
        self._parts: list[str] = []
        self.cells: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"td", "th"}:
            self._in_cell += 1
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._in_cell:
            self._in_cell -= 1
            text = normalize_text("".join(self._parts))
            self.cells.append(text)
            self._parts = []


def normalize_text(text: str | None) -> str:
    return " ".join(html.unescape(text or "").split()).lower()


def table_cells(content: str | None) -> list[str]:
    parser = _CellParser()
    parser.feed(content or "")
    return parser.cells


def table_exact_cell_counts(ref_blocks: list[dict[str, Any]],
                            pred_blocks: list[dict[str, Any]]) -> tuple[int, int]:
    ref_cells: list[str] = []
    pred_cells: list[str] = []
    for ref, pred in zip(ref_blocks, pred_blocks):
        if ref.get("type") != "table":
            continue
        ref_cells.extend(table_cells(ref.get("content")))
        pred_cells.extend(table_cells(pred.get("content")))
    matched = 0
    for i, cell in enumerate(ref_cells):
        if i < len(pred_cells) and pred_cells[i] == cell:
            matched += 1
    return matched, len(ref_cells)


def table_exact_cell_recall(ref_blocks: list[dict[str, Any]],
                            pred_blocks: list[dict[str, Any]]) -> float | None:
    matched, total = table_exact_cell_counts(ref_blocks, pred_blocks)
    if total == 0:
        return None
    return matched / total


def compare_blocks(ref: list[dict[str, Any]],
                   pred: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(ref), len(pred))
    type_matches = [
        1.0 if ref[i].get("type") == pred[i].get("type") else 0.0
        for i in range(n)
    ]
    bbox_ious = [
        bbox_iou(list(ref[i].get("bbox") or [0, 0, 0, 0]),
                 list(pred[i].get("bbox") or [0, 0, 0, 0]))
        for i in range(n)
    ]
    content_f1s = [
        token_f1(ref[i].get("content"), pred[i].get("content"))
        for i in range(n)
    ]
    table_recall = table_exact_cell_recall(ref, pred)
    return {
        "ref_blocks": len(ref),
        "pred_blocks": len(pred),
        "block_count_exact": len(ref) == len(pred),
        "ordered_type_accuracy": mean(type_matches) if type_matches else 1.0,
        "ordered_mean_bbox_iou": mean(bbox_ious) if bbox_ious else 1.0,
        "ordered_median_bbox_iou": median(bbox_ious) if bbox_ious else 1.0,
        "mean_content_token_f1": mean(content_f1s) if content_f1s else 1.0,
        "table_exact_cell_recall": table_recall,
        "ref_types": [b.get("type") for b in ref],
        "pred_types": [b.get("type") for b in pred],
    }


def compare_pages(
    pages: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]]
) -> dict[str, Any]:
    per_page: list[dict[str, Any]] = []
    type_matches: list[float] = []
    bbox_ious: list[float] = []
    content_f1s: list[float] = []
    block_count_exact_pages = 0
    total_ref_blocks = 0
    total_pred_blocks = 0
    table_pages = 0
    table_matched = 0
    table_total = 0

    for page, ref, pred in pages:
        metrics = compare_blocks(ref, pred)
        per_page.append({"page": page, **metrics})
        if metrics["block_count_exact"]:
            block_count_exact_pages += 1
        total_ref_blocks += len(ref)
        total_pred_blocks += len(pred)
        n = min(len(ref), len(pred))
        for i in range(n):
            type_matches.append(
                1.0 if ref[i].get("type") == pred[i].get("type") else 0.0
            )
            bbox_ious.append(
                bbox_iou(
                    list(ref[i].get("bbox") or [0, 0, 0, 0]),
                    list(pred[i].get("bbox") or [0, 0, 0, 0]),
                )
            )
            content_f1s.append(token_f1(ref[i].get("content"), pred[i].get("content")))
        matched, total = table_exact_cell_counts(ref, pred)
        if total:
            table_pages += 1
            table_matched += matched
            table_total += total

    page_count = len(pages)
    return {
        "page_count": page_count,
        "total_ref_blocks": total_ref_blocks,
        "total_pred_blocks": total_pred_blocks,
        "total_ordered_blocks": len(type_matches),
        "block_count_exact_pages": block_count_exact_pages,
        "block_count_exact_rate": (
            block_count_exact_pages / page_count if page_count else 1.0
        ),
        "ordered_type_accuracy": mean(type_matches) if type_matches else 1.0,
        "ordered_mean_bbox_iou": mean(bbox_ious) if bbox_ious else 1.0,
        "ordered_median_bbox_iou": median(bbox_ious) if bbox_ious else 1.0,
        "mean_content_token_f1": mean(content_f1s) if content_f1s else 1.0,
        "table_pages": table_pages,
        "table_exact_cells_matched": table_matched,
        "table_exact_cells_total": table_total,
        "table_exact_cell_recall": (
            table_matched / table_total if table_total else None
        ),
        "pages": per_page,
    }


def load_blocks_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON block list")
    return data


def load_blocks_from_pages_jsonl(path: Path, page: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("page") == page and row.get("status", "ok") == "ok":
                result = row.get("result")
                if not isinstance(result, list):
                    raise ValueError(f"page {page} in {path} has no result list")
                return result
    raise ValueError(f"page {page} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-json", type=Path)
    parser.add_argument("--ref-pages-jsonl", type=Path)
    parser.add_argument("--page", type=int)
    parser.add_argument("--pages")
    parser.add_argument("--pred-json", type=Path)
    parser.add_argument("--pred-json-template")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if bool(args.ref_json) == bool(args.ref_pages_jsonl):
        raise SystemExit("provide exactly one of --ref-json or --ref-pages-jsonl")
    if args.pages or args.pred_json_template:
        if not args.ref_pages_jsonl:
            raise SystemExit("--pages requires --ref-pages-jsonl")
        if not args.pages or not args.pred_json_template:
            raise SystemExit("provide both --pages and --pred-json-template")
        page_numbers = [
            int(part.strip())
            for part in args.pages.split(",")
            if part.strip()
        ]
        pages = [
            (
                page,
                load_blocks_from_pages_jsonl(args.ref_pages_jsonl, page),
                load_blocks_json(Path(args.pred_json_template.format(page=page))),
            )
            for page in page_numbers
        ]
        metrics = compare_pages(pages)
        text = json.dumps(metrics, indent=2, ensure_ascii=False)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
        return

    if not args.pred_json:
        raise SystemExit("--pred-json is required")
    if args.ref_pages_jsonl and args.page is None:
        raise SystemExit("--ref-pages-jsonl requires --page")

    ref = (load_blocks_json(args.ref_json) if args.ref_json
           else load_blocks_from_pages_jsonl(args.ref_pages_jsonl, args.page))
    pred = load_blocks_json(args.pred_json)
    metrics = compare_blocks(ref, pred)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
