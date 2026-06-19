#!/usr/bin/env python3
from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mu_compare_outputs


class MuCompareOutputsTest(unittest.TestCase):
    def test_bbox_iou_uses_intersection_over_union(self) -> None:
        self.assertAlmostEqual(
            mu_compare_outputs.bbox_iou([0, 0, 1, 1], [0.5, 0.5, 1, 1]),
            0.25,
        )

    def test_token_f1_normalizes_text(self) -> None:
        self.assertAlmostEqual(
            mu_compare_outputs.token_f1("Hello,  world!", "hello world"),
            2 / 3,
        )

    def test_compare_blocks_reports_layout_content_and_table_metrics(self) -> None:
        ref = [
            {
                "type": "table",
                "bbox": [0, 0, 1, 1],
                "content": "<table><tr><td>A</td><td>B</td></tr></table>",
            },
            {"type": "footer", "bbox": [0, 0.9, 1, 1], "content": "NASA"},
        ]
        pred = [
            {
                "type": "table",
                "bbox": [0, 0, 1, 1],
                "content": "<table><tr><td>A</td><td>C</td></tr></table>",
            },
            {"type": "footer", "bbox": [0, 0.9, 1, 1], "content": "NASA"},
        ]

        metrics = mu_compare_outputs.compare_blocks(ref, pred)

        self.assertEqual(metrics["block_count_exact"], True)
        self.assertEqual(metrics["ordered_type_accuracy"], 1.0)
        self.assertEqual(metrics["ordered_mean_bbox_iou"], 1.0)
        self.assertLess(metrics["mean_content_token_f1"], 1.0)
        self.assertEqual(metrics["table_exact_cell_recall"], 0.5)

    def test_compare_pages_aggregates_weighted_metrics(self) -> None:
        pages = [
            (
                10,
                [
                    {
                        "type": "table",
                        "bbox": [0, 0, 1, 1],
                        "content": "<table><tr><td>A</td><td>B</td></tr></table>",
                    },
                    {"type": "footer", "bbox": [0, 0, 1, 1], "content": "NASA"},
                ],
                [
                    {
                        "type": "table",
                        "bbox": [0, 0, 1, 1],
                        "content": "<table><tr><td>A</td><td>C</td></tr></table>",
                    },
                    {"type": "footer", "bbox": [0, 0, 1, 1], "content": "NASA"},
                ],
            ),
            (
                11,
                [{"type": "text", "bbox": [0, 0, 1, 1], "content": "Alpha"}],
                [
                    {"type": "title", "bbox": [0, 0, 1, 1], "content": "Alpha"},
                    {"type": "footer", "bbox": [0, 0, 1, 1], "content": "extra"},
                ],
            ),
        ]

        metrics = mu_compare_outputs.compare_pages(pages)

        self.assertEqual(metrics["page_count"], 2)
        self.assertEqual(metrics["block_count_exact_pages"], 1)
        self.assertEqual(metrics["block_count_exact_rate"], 0.5)
        self.assertEqual(metrics["total_ordered_blocks"], 3)
        self.assertAlmostEqual(metrics["ordered_type_accuracy"], 2 / 3)
        self.assertLess(metrics["mean_content_token_f1"], 1.0)
        self.assertEqual(metrics["table_pages"], 1)
        self.assertEqual(metrics["table_exact_cell_recall"], 0.5)
        self.assertEqual(metrics["pages"][1]["page"], 11)

    def test_load_page_pairs_supports_json_templates(self) -> None:
        tmp = Path("/tmp/mu-compare-outputs-test")
        tmp.mkdir(exist_ok=True)
        (tmp / "ref_0001.json").write_text(
            '[{"type":"text","bbox":[0,0,1,1],"content":"hello"}]\n',
            encoding="utf-8",
        )
        (tmp / "pred_0001.json").write_text(
            '[{"type":"text","bbox":[0,0,1,1],"content":"hello"}]\n',
            encoding="utf-8",
        )

        pairs = mu_compare_outputs.load_page_pairs(
            [1],
            ref_json_template=str(tmp / "ref_{page:04d}.json"),
            pred_json_template=str(tmp / "pred_{page:04d}.json"),
        )

        self.assertEqual(pairs[0][0], 1)
        self.assertEqual(pairs[0][1][0]["content"], "hello")
        self.assertEqual(pairs[0][2][0]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
