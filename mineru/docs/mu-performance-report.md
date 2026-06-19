# mineru/mu.c Performance Test Report

Date: 2026-06-17
Branch: `codex/mineru-mu-engine`
Repo commit base: `91bafb5` with local uncommitted MU work

This report records the current performance and parity baseline for
`mineru/mu.c` against the local Transformers implementation in
`/Users/will/github/mineru-model`.

## Summary

The native `mu` path now matches the sampled Transformers outputs on the tested
pages, but it is not performance-competitive yet.

| Metric | Result |
| --- | ---: |
| Sampled pages | 10 |
| Exact block-count pages | 10 / 10 |
| Ordered block type accuracy | 100.00% |
| Mean bbox IoU | 0.9877 |
| Median bbox IoU | 1.0000 |
| Mean content token F1 | 1.0000 |
| Table exact cell recall | 1.0000 on 6 table pages |
| Transformers/MPS total time | 386.98 s |
| Transformers/MPS mean time | 38.70 s/page |
| Native `mu` total time | 883.43 s |
| Native `mu` mean time | 88.34 s/page |
| Current speed gap | Native `mu` is about 2.3x slower |

Interpretation:

- Correctness is already strong for this smoke corpus: layout block counts,
  ordered block types, table structure, and extracted content match the
  Transformers reference on all sampled pages.
- Performance is expectedly behind: the current `mu` implementation is a
  CPU/Accelerate-oriented correctness path, while the reference uses PyTorch MPS
  kernels on Apple GPU.
- The next meaningful performance step is Metal coverage for the vision tower
  and dense decoder matmuls, not small C-level refactoring.

## Test Environment

| Item | Value |
| --- | --- |
| Machine | MacBook Pro, Mac17,2 |
| Chip | Apple M5, 10 cores |
| Memory | 16 GB |
| OS | macOS 27.0, build 26A5353q |
| Python reference | Python 3.13.13 |
| PyTorch reference | 2.12.0 |
| Reference accelerator | MPS available |
| Native backend | `MU_BACKEND_CPU` with Accelerate/CBLAS calls |

## Test Corpus

Source PDF:

```text
/Users/will/github/mineru-model/testdata/nasa_systems_engineering_handbook_rev2.pdf
```

Sampled pages:

```text
224, 234, 237, 241, 244, 247, 258, 281, 303, 334
```

The sample intentionally mixes table-heavy glossary pages and normal text pages:

- Table pages: 224, 234, 237, 241, 244, 247
- Text/title pages: 258, 281, 303, 334

The comparison used 120dpi page renders. The reference timing rows were selected
from the local `mineru-model` MPS runs at `1020x1320` page size; page 224 also
has a dedicated 120dpi reference run.

## Accuracy Results

Raw metrics artifact:

```text
/tmp/mu-compare-10/metrics.json
```

Aggregate metrics:

| Metric | Value |
| --- | ---: |
| Page count | 10 |
| Total ordered blocks | 34 |
| Greedy matched blocks | 34 |
| Block count exact rate | 1.0000 |
| Ordered type accuracy | 1.0000 |
| Ordered mean bbox IoU | 0.9877 |
| Ordered median bbox IoU | 1.0000 |
| Mean content token F1 | 1.0000 |
| Mean content sequence ratio | 1.0000 |

Per-page metrics:

| Page | Blocks ref/pred | Type acc | BBox IoU | Content F1 |
| ---: | ---: | ---: | ---: | ---: |
| 224 | 3/3 | 1.000 | 0.9444 | 1.000 |
| 234 | 3/3 | 1.000 | 1.0000 | 1.000 |
| 237 | 3/3 | 1.000 | 0.9815 | 1.000 |
| 241 | 3/3 | 1.000 | 0.9989 | 1.000 |
| 244 | 3/3 | 1.000 | 1.0000 | 1.000 |
| 247 | 3/3 | 1.000 | 0.9449 | 1.000 |
| 258 | 4/4 | 1.000 | 0.9965 | 1.000 |
| 281 | 4/4 | 1.000 | 1.0000 | 1.000 |
| 303 | 4/4 | 1.000 | 0.9971 | 1.000 |
| 334 | 4/4 | 1.000 | 0.9997 | 1.000 |

Table details:

| Page | Rows ref/pred | Cells ref/pred | Exact cell recall |
| ---: | ---: | ---: | ---: |
| 224 | 11/11 | 22/22 | 1.000 |
| 234 | 7/7 | 14/14 | 1.000 |
| 237 | 9/9 | 18/18 | 1.000 |
| 241 | 6/6 | 12/12 | 1.000 |
| 244 | 9/9 | 18/18 | 1.000 |
| 247 | 10/10 | 20/20 | 1.000 |

## Timing Results

Reference timing sources:

```text
/Users/will/github/mineru-model/runs/nasa_systems_engineering_handbook_rev2_full_mps/pages.jsonl
/Users/will/github/mineru-model/runs/nasa_systems_engineering_handbook_rev2_page224_mps_120dpi_test/pages.jsonl
```

Native timing artifacts:

```text
/tmp/mu-compare-10/run_summary.json
/tmp/mu-compare-10/mu_page_*.json
```

Transformers/MPS per-page total time:

| Page | Transformers total s | Blocks |
| ---: | ---: | ---: |
| 224 | 52.93 | 3 |
| 234 | 51.00 | 3 |
| 237 | 51.05 | 3 |
| 241 | 51.36 | 3 |
| 244 | 51.18 | 3 |
| 247 | 43.06 | 3 |
| 258 | 26.33 | 4 |
| 281 | 19.19 | 4 |
| 303 | 17.56 | 4 |
| 334 | 23.31 | 4 |

Native `mu` per-page total time from the visible run summary:

| Page | Native total s | Blocks | Note |
| ---: | ---: | ---: | --- |
| 224 | reused | 3 | output artifact reused in this summary |
| 234 | 103.19 | 3 |  |
| 237 | 105.52 | 3 |  |
| 241 | 110.82 | 3 |  |
| 244 | 107.56 | 3 |  |
| 247 | 106.38 | 3 |  |
| 258 | 60.46 | 4 |  |
| 281 | 61.20 | 4 |  |
| 303 | 63.11 | 4 |  |
| 334 | 65.06 | 4 |  |

Timing notes:

- The 10-page native summary used for the headline is 883.43s total,
  88.34s/page mean.
- The currently visible `run_summary.json` marks page 224 as reused. The 9
  freshly measured native rows sum to 783.31s, or 87.03s/page.
- Excluding page 224, the same 9 pages take 334.05s in Transformers/MPS and
  783.31s in native `mu`, or about 2.35x slower. This agrees with the headline
  conclusion: the current native path is roughly 2.3x slower.

## Bottleneck Assessment

The current gap is structural:

1. Native `mu` is still optimized for correctness and trace parity.
2. Dense BF16 matmuls and attention are not yet fully moved to Metal.
3. The page pipeline still spends time in repeated generation passes for layout
   and content extraction.
4. Transformers benefits from mature PyTorch MPS scheduling and fused kernels.

Most likely optimization order:

1. Add Metal kernels for Qwen2-VL vision tower dense matmuls, norms, attention,
   and merge projection.
2. Move decoder dense matmuls and attention to Metal.
3. Add KV-cache decode instead of repeated full-prefill generation.
4. Batch same-type content crops where the MinerU prompt allows it.
5. Keep CPU path as a deterministic reference and regression harness.

## Current Conclusion

`mineru/mu.c` is ready as a correctness baseline for MinerU2.5-Pro page parsing
experiments. It is not yet the faster production path. The native engine should
now be optimized around Metal execution, with this report serving as the first
baseline to beat.

## Metal Backend Checkpoint

Date: 2026-06-19
Branch: `codex/mineru-metal-backend`
Commits: `60dc6ee`, `50cf405`, `c3c8312`

The first full-page Metal correctness bridge is now available behind
`--backend metal --no-cpu-fallback`. CPU remains the reference backend and is
still the default. The current Metal path is intentionally conservative: it
keeps C-side orchestration, uses unoptimized row-wise vision attention kernels,
and still pays repeated full-prefill generation costs in the text decoder.

Fresh validation before this checkpoint:

```text
make -B mu
make mu-test
./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_metal_vision_encode_smoke.py
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_metal_vision_smoke.py
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_metal_page_smoke.py
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_compare_backends.py
git diff --check
```

The backend comparison smoke used `--compare-backends --skip-content
--max-new-tokens 4` on `/Users/will/github/mineru-model/sample_page.png`; CPU
and Metal produced identical JSON and Metal reported zero fallback.

Stage timing artifacts:

```text
/tmp/mu-benchmark-cpu-page224-smoke.json
/tmp/mu-benchmark-metal-page224-smoke.json
/tmp/mu-benchmark-cpu-page224-layout.json
/tmp/mu-benchmark-cpu-10-smoke.json
/tmp/mu-benchmark-metal-10-smoke.json
/tmp/mu-benchmark-cpu-page224-token1.json
/tmp/mu-benchmark-metal-page224-token1.json
/tmp/mu-benchmark-metal-page224-layout128.json
```

10-page quick E2E smoke, `--max-new-tokens 4 --skip-content`:

| Backend | Total s | Mean s/page | Completed pages | CPU fallback rows |
| --- | ---: | ---: | ---: | ---: |
| CPU reference | 506.40 | 50.64 | 10 / 10 | 0 |
| Metal no-fallback | 1568.66 | 156.87 | 10 / 10 | 0 |

Per-page quick timing:

| Page | CPU s | Metal s | Metal fallback |
| ---: | ---: | ---: | ---: |
| 224 | 55.23 | 170.24 | 0 |
| 234 | 53.05 | 133.15 | 0 |
| 237 | 50.27 | 131.02 | 0 |
| 241 | 49.46 | 132.80 | 0 |
| 244 | 48.90 | 131.81 | 0 |
| 247 | 49.23 | 197.38 | 0 |
| 258 | 49.90 | 201.94 | 0 |
| 281 | 48.28 | 202.46 | 0 |
| 303 | 48.98 | 129.65 | 0 |
| 334 | 53.11 | 138.21 | 0 |

Page 224 quick smoke, `--max-new-tokens 4 --skip-content`:

| Backend | Seconds | Blocks | CPU fallback rows |
| --- | ---: | ---: | ---: |
| CPU reference | 50.56 | 0 | 0 |
| Metal no-fallback | 194.05 | 0 | 0 |

Page 224 1-token diagnostic, `--max-new-tokens 1 --skip-content`:

| Backend | Seconds | Blocks | CPU fallback rows |
| --- | ---: | ---: | ---: |
| CPU reference | 52.29 | 0 | 0 |
| Metal no-fallback | 137.26 | 0 | 0 |

Page 224 layout-only parity, `--max-new-tokens 128 --skip-content`:

| Backend | Seconds | Blocks | Ordered types | CPU fallback rows |
| --- | ---: | ---: | --- | ---: |
| CPU reference | 54.09 | 3 | table, footer, page_number |
| Metal no-fallback | 610.99 | 3 | table, footer, page_number | 0 |

Interpretation:

- Metal no-fallback is functionally wired through the 10 sampled pages, but the
  current correctness bridge is slower than CPU on this quick benchmark by about
  3.1x.
- The 4-token smoke is useful for end-to-end process timing and fallback
  detection across the sampled corpus, but not for accuracy, because it stops
  before layout blocks are emitted.
- The 128-token page224 layout-only run is the first page-level accuracy result
  for the pure Metal path: Metal no-fallback matches CPU block count and ordered
  block types (`table`, `footer`, `page_number`) with zero fallback. It is still
  about 11.3x slower than CPU for this page.
- The 1-token diagnostic shows that most current Metal time is already spent
  before token generation has much room to accumulate. The next optimization
  target is therefore full-page vision encode: keep intermediate tensors and
  weights in reusable Metal buffers across the 32 vision blocks, then revisit
  KV-cache decode.
- The next performance work should target reusable Metal buffers, fused vision
  attention, dense-kernel batching, and KV-cache decode. Until then, Metal
  numbers should be reported as correctness-bridge numbers, not production
  acceleration.
