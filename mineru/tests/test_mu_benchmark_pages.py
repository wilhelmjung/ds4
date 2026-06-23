#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mu_benchmark_pages


class MuBenchmarkPagesTest(unittest.TestCase):
    def test_run_one_saves_successful_json_output(self) -> None:
        blocks = [
            {
                "type": "table",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "angle": 0,
                "content": "<table></table>",
            }
        ]
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(blocks),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with mock.patch.object(mu_benchmark_pages.subprocess, "run",
                                   return_value=completed):
                row = mu_benchmark_pages.run_one(
                    "metal",
                    Path("/tmp/page.png"),
                    max_new_tokens=128,
                    skip_content=False,
                    timeout=10,
                    output_dir=out_dir,
                    page=224,
                )

            self.assertEqual(row["blocks"], 1)
            self.assertEqual(row["types"], ["table"])
            self.assertIn("output_json", row)
            saved = Path(row["output_json"])
            self.assertEqual(saved.parent, out_dir)
            self.assertEqual(saved.name, "metal_page_0224.json")
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), blocks)

    def test_run_one_passes_content_token_limit_in_environment(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        with mock.patch.object(mu_benchmark_pages.subprocess, "run",
                               return_value=completed) as run_mock:
            row = mu_benchmark_pages.run_one(
                "cpu",
                Path("/tmp/page.png"),
                max_new_tokens=128,
                skip_content=False,
                timeout=10,
                content_max_new_tokens=512,
            )

        self.assertEqual(row["content_max_new_tokens"], 512)
        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env["MU_CONTENT_MAX_NEW_TOKENS"], "512")

    def test_run_one_records_stage_timings_from_stderr(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="[]",
            stderr=(
                "mu_timing stage=layout_preprocess seconds=0.125\n"
                "mu_timing stage=vision_encode seconds=4.5\n"
                "mu_timing stage=content_region_generate seconds=1.25\n"
                "mu_timing stage=content_region_generate seconds=2.5\n"
            ),
        )
        with mock.patch.object(mu_benchmark_pages.subprocess, "run",
                               return_value=completed) as run_mock:
            row = mu_benchmark_pages.run_one(
                "metal",
                Path("/tmp/page.png"),
                max_new_tokens=1,
                skip_content=True,
                timeout=10,
                timing=True,
            )

        self.assertEqual(
            row["stage_timings"],
            {
                "layout_preprocess": 0.125,
                "vision_encode": 4.5,
                "content_region_generate": 3.75,
            },
        )
        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env["MU_TIMING"], "1")

    def test_run_one_records_profile_lines_from_stderr(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="[]",
            stderr=(
                "mu_profile stage=vision_attn_shape layer=3 rows=5476\n"
                "mu_profile stage=vision_attn_shape path=flash rows=5476 "
                "threadgroups=172x16x1 threads=32x1x1 "
                "query_tile_rows=32 key_tile_rows=32 heads=16\n"
            ),
        )
        with mock.patch.object(mu_benchmark_pages.subprocess, "run",
                               return_value=completed):
            row = mu_benchmark_pages.run_one(
                "metal",
                Path("/tmp/page.png"),
                max_new_tokens=1,
                skip_content=True,
                timeout=10,
            )

        self.assertEqual(
            row["profiles"]["vision_attn_shape"],
            [
                {"layer": "3", "rows": "5476"},
                {
                    "path": "flash",
                    "rows": "5476",
                    "threadgroups": "172x16x1",
                    "threads": "32x1x1",
                    "query_tile_rows": "32",
                    "key_tile_rows": "32",
                    "heads": "16",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
