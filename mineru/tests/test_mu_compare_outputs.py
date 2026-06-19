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


if __name__ == "__main__":
    unittest.main()
