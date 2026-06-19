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
Measurement code commits: `60dc6ee` through `a8d8d43`

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
/tmp/mu-benchmark-cpu-10-layout128.json
/tmp/mu-benchmark-metal-page224-layout128.json
/tmp/mu-benchmark-metal-layout128-batch1.json
/tmp/mu-benchmark-metal-layout128-batch2.json
/tmp/mu-benchmark-metal-layout128-batch3.json
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

10-page layout-only parity, `--max-new-tokens 128 --skip-content`:

| Backend | Total s | Mean s/page | Completed pages | CPU fallback rows |
| --- | ---: | ---: | ---: | ---: |
| CPU reference | 511.64 | 51.16 | 10 / 10 | 0 |
| Metal no-fallback | 7435.68 | 743.57 | 10 / 10 | 0 |

| Page | CPU s | Metal s | CPU blocks/types | Metal blocks/types | Match | Metal fallback |
| ---: | ---: | ---: | --- | --- | ---: | ---: |
| 224 | 52.44 | 610.99 | 3 table/footer/page_number | 3 table/footer/page_number | yes | 0 |
| 234 | 51.02 | 788.66 | 3 table/footer/page_number | 3 table/footer/page_number | yes | 0 |
| 237 | 50.92 | 742.55 | 3 table/footer/page_number | 3 table/footer/page_number | yes | 0 |
| 241 | 50.49 | 574.56 | 3 table/footer/page_number | 3 table/footer/page_number | yes | 0 |
| 244 | 50.10 | 583.62 | 3 table/footer/page_number | 3 table/footer/page_number | yes | 0 |
| 247 | 50.44 | 569.31 | 3 table/footer/page_number | 3 table/footer/page_number | yes | 0 |
| 258 | 51.38 | 748.50 | 4 title/text/footer/page_number | 4 title/text/footer/page_number | yes | 0 |
| 281 | 51.76 | 739.76 | 4 title/text/footer/page_number | 4 title/text/footer/page_number | yes | 0 |
| 303 | 51.51 | 793.89 | 4 title/text/footer/page_number | 4 title/text/footer/page_number | yes | 0 |
| 334 | 51.59 | 1283.85 | 4 text/text/footer/page_number | 4 text/text/footer/page_number | yes | 0 |

Interpretation:

- Metal no-fallback is functionally wired through the 10 sampled pages, but the
  current correctness bridge is slower than CPU on this quick benchmark by about
  3.1x.
- The 4-token smoke is useful for end-to-end process timing and fallback
  detection across the sampled corpus, but not for accuracy, because it stops
  before layout blocks are emitted.
- The 128-token layout-only run completes all 10 sampled pages in pure Metal
  no-fallback mode. Metal matches CPU block count and ordered block types on all
  10 pages with zero fallback. It is still about 14.5x slower than CPU on this
  layout-only benchmark.
- The 1-token diagnostic shows that most current Metal time is already spent
  before token generation has much room to accumulate. The next optimization
  target is therefore full-page vision encode: keep intermediate tensors and
  weights in reusable Metal buffers across the 32 vision blocks, then revisit
  KV-cache decode.
- The next performance work should target reusable Metal buffers, fused vision
  attention, dense-kernel batching, and KV-cache decode. Until then, Metal
  numbers should be reported as correctness-bridge numbers, not production
  acceleration.

## Full-content Page 224 E2E Checkpoint

Date: 2026-06-19
Branch: `codex/mineru-metal-backend`

This checkpoint extends the pure Metal validation from layout-only generation to
full page extraction on page 224. The run uses Metal with CPU fallback disabled:

```text
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --pages 224 \
  --max-new-tokens 128 \
  --content-max-new-tokens 512 \
  --timeout 7200 \
  --resume \
  --out /tmp/mu-benchmark-metal-page224-fullcontent-content512.json \
  --save-output-dir /tmp/mu-fullcontent-page224-content512
```

The matching CPU reference command used the same layout/content token limits and
saved `/tmp/mu-fullcontent-page224-content512/cpu_page_0224.json`.

Reference artifacts:

```text
/Users/will/github/mineru-model/runs/nasa_systems_engineering_handbook_rev2_page224_mps_120dpi_test/pages.jsonl
/tmp/mu-benchmark-cpu-page224-fullcontent-content512.json
/tmp/mu-benchmark-metal-page224-fullcontent-content512.json
/tmp/mu-fullcontent-page224-content512/cpu_page_0224.json
/tmp/mu-fullcontent-page224-content512/metal_page_0224.json
/tmp/mu-fullcontent-page224-content512/cpu-vs-metal.metrics.json
/tmp/mu-fullcontent-page224-content512/transformers120-vs-cpu.metrics.json
/tmp/mu-fullcontent-page224-content512/transformers120-vs-metal.metrics.json
```

Performance:

| Backend | Seconds | Blocks/types | CPU fallback rows |
| --- | ---: | --- | ---: |
| Transformers/MPS 120dpi reference | 52.93 | 3 table/footer/page_number | n/a |
| CPU reference | 109.63 | 3 table/footer/page_number | 0 |
| Metal no-fallback | 3437.83 | 3 table/footer/page_number | 0 |

Speed ratios:

| Comparison | Ratio |
| --- | ---: |
| CPU / Transformers | 2.07x slower |
| Metal / CPU | 31.36x slower |
| Metal / Transformers | 64.95x slower |

Accuracy, CPU versus Metal:

| Metric | Value |
| --- | ---: |
| Block count exact | true |
| Ordered type accuracy | 1.0000 |
| Ordered mean bbox IoU | 1.0000 |
| Ordered median bbox IoU | 1.0000 |
| Mean content token F1 | 1.0000 |
| Table exact cell recall | 1.0000 |

Accuracy, Transformers/MPS 120dpi versus Metal:

| Metric | Value |
| --- | ---: |
| Block count exact | true |
| Ordered type accuracy | 1.0000 |
| Ordered mean bbox IoU | 0.9444 |
| Ordered median bbox IoU | 0.9412 |
| Mean content token F1 | 1.0000 |
| Table exact cell recall | 1.0000 |

Diagnostic note:

- With `--content-max-new-tokens` unset and `--max-new-tokens 128`, CPU and
  Metal are still exactly identical, but the table content is truncated after
  the third row. Against the 512-token Transformers reference, that diagnostic
  run reports content F1 `0.8302` and table cell recall `0.3182`. It is useful
  for CPU/Metal parity and fallback detection, but not for content-completeness
  accuracy.
- With `--content-max-new-tokens 512`, Metal matches CPU exactly and matches the
  Transformers page-224 table/footer/page-number content. This is the first
  full-content pure Metal no-fallback E2E validation.

Interpretation:

- Pure Metal execution is now functionally complete for this page-level
  full-content case: vision encode, layout generation, crop extraction, table
  recognition, footer recognition, page-number recognition, JSON writing, and
  fallback detection all run with `--no-cpu-fallback`.
- Performance is not production-ready. The 1-token and layout-only diagnostics
  already pointed at vision encode and repeated host/device buffer work; the
  content512 result confirms that full-content extraction amplifies the same
  bottleneck. The next implementation target remains persistent Metal buffers
  for weights/intermediates and fewer per-kernel host-device copies, followed by
  KV-cache decode.

## Stage Timing Checkpoint

Date: 2026-06-19
Branch: `codex/mineru-metal-backend`

The benchmark harness now supports `--timing`, which sets `MU_TIMING=1` for the
`mu` subprocess and records `mu_timing stage=... seconds=...` rows from stderr
into each benchmark row as `stage_timings`.

Timing smoke commands:

```text
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend cpu \
  --pages 224 \
  --max-new-tokens 1 \
  --skip-content \
  --timing \
  --timeout 900 \
  --out /tmp/mu-benchmark-cpu-page224-token1-timing.json

/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --pages 224 \
  --max-new-tokens 1 \
  --skip-content \
  --timing \
  --timeout 1200 \
  --out /tmp/mu-benchmark-metal-page224-token1-timing.json
```

Page 224, 1-token layout smoke stage timings:

| Stage | CPU s | Metal s | Metal / CPU |
| --- | ---: | ---: | ---: |
| layout_preprocess | 0.0273 | 0.0393 | 1.44x |
| layout_patch_embed | 0.0227 | 0.0261 | 1.15x |
| layout_rotary | 0.0001 | 0.0001 | 1.03x |
| layout_vision_encode | 39.5267 | 101.0340 | 2.56x |
| layout_prompt_tokenize | 3.7702 | 3.8659 | 1.03x |
| layout_generate | 4.6844 | 7.6832 | 1.64x |
| layout_decode_parse | 0.0000 | 0.0000 | 0.25x |
| result_build | 0.0000 | 0.0000 | 3.50x |
| page_total | 48.0315 | 112.6486 | 2.35x |

Interpretation:

- The 1-token timing smoke isolates the front half of the pipeline. It is not a
  full-content benchmark, but it directly shows where the fixed per-page cost is
  concentrated.
- Full-page `layout_vision_encode` dominates both CPU and Metal time. On this
  run, Metal spends `101.03s` in layout vision encode versus `39.53s` on CPU.
- Prompt tokenization and result building are not meaningful bottlenecks at this
  stage.
- The next performance experiment should instrument and reduce Metal buffer
  churn inside vision encode before broadening full-content runs.

## Full-content Stage Timing Checkpoint

Date: 2026-06-19
Branch: `codex/mineru-metal-backend`

After the 1-token timing smoke isolated fixed page costs, a page 224
full-content timing run was executed with layout `--max-new-tokens 128` and
`--content-max-new-tokens 128`. This run is still a truncation diagnostic for
content completeness, but it exercises the full layout plus three content-block
passes and preserves CPU/Metal exact parity.

Artifacts:

```text
/tmp/mu-benchmark-cpu-page224-fullcontent128-timing.json
/tmp/mu-benchmark-metal-page224-fullcontent128-timing.json
/tmp/mu-fullcontent128-timing-page224/cpu_page_0224.json
/tmp/mu-fullcontent128-timing-page224/metal_page_0224.json
/tmp/mu-fullcontent128-timing-page224/cpu-vs-metal.metrics.json
```

Command shape:

```text
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --pages 224 \
  --max-new-tokens 128 \
  --content-max-new-tokens 128 \
  --timing \
  --timeout 3600 \
  --out /tmp/mu-benchmark-metal-page224-fullcontent128-timing.json \
  --save-output-dir /tmp/mu-fullcontent128-timing-page224
```

Accuracy, CPU versus Metal:

| Metric | Value |
| --- | ---: |
| Block count exact | true |
| Ordered type accuracy | 1.0000 |
| Ordered mean bbox IoU | 1.0000 |
| Ordered median bbox IoU | 1.0000 |
| Mean content token F1 | 1.0000 |
| Table exact cell recall | 1.0000 |

Page 224 full-content128 stage timings:

| Stage | CPU s | Metal s | Metal / CPU |
| --- | ---: | ---: | ---: |
| layout_vision_encode | 49.1435 | 135.5313 | 2.76x |
| layout_generate | 8.2231 | 494.9236 | 60.19x |
| content_region_vision_encode | 32.2100 | 70.3365 | 2.18x |
| content_region_generate | 11.9075 | 680.4337 | 57.14x |
| content_total | 44.2390 | 750.9101 | 16.97x |
| page_total | 105.5664 | 1385.3995 | 13.12x |

Interpretation:

- The earlier 1-token timing smoke correctly identified full-page vision encode
  as the dominant fixed per-page cost.
- In full-content mode, repeated full-prefill generation dominates Metal wall
  time: `layout_generate` plus `content_region_generate` accounts for about
  `1175.36s` of the `1385.40s` page total.
- Vision encode still matters: layout plus content-region vision encode accounts
  for about `205.87s` on Metal.
- The next optimization split should therefore be explicit: first keep reducing
  vision encode buffer churn, but the largest full-content win requires a
  KV-cache decode path or equivalent removal of repeated full-prefill generation.

## KV-cache Decode Checkpoint

Date: 2026-06-19
Branch: `codex/mineru-metal-backend`

This checkpoint replaces the Metal generation loop's repeated full-prefill
calls with a cache-backed decode path. The prefill still uses the existing
Metal sequence kernels, but now records per-layer K/V cache rows. Subsequent
decode tokens use one-token Metal dense/norm/MLP kernels plus the new
`mu_text_attn_cached` Metal attention kernel. CPU remains the default backend
and precision reference.

New validation gates:

```text
make mu-test
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_metal_text_generation_smoke.py
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_metal_layout_generation_smoke.py
```

The generation smoke tests now require `mu metal stage: text_cached_attn` in
stderr, so they distinguish cached decode from the older repeated full-prefill
path.

Artifacts:

```text
/tmp/mu-benchmark-cpu-page224-layout128-kvcache-baseline.json
/tmp/mu-benchmark-metal-page224-layout128-kvcache.json
/tmp/mu-benchmark-cpu-page224-fullcontent128-kvcache-baseline.json
/tmp/mu-benchmark-metal-page224-fullcontent128-kvcache.json
/tmp/mu-fullcontent128-kvcache-page224/cpu_page_0224.json
/tmp/mu-fullcontent128-kvcache-page224/metal_page_0224.json
/tmp/mu-fullcontent128-kvcache-page224/cpu-vs-metal.metrics.json
```

Page 224 layout-only, `--max-new-tokens 128 --skip-content --timing`:

| Stage | CPU s | Metal s | Metal / CPU |
| --- | ---: | ---: | ---: |
| layout_vision_encode | 40.9909 | 113.6929 | 2.77x |
| layout_generate | 8.8269 | 29.0975 | 3.30x |
| page_total | 54.2621 | 147.0423 | 2.71x |

Page 224 full-content128, `--max-new-tokens 128
--content-max-new-tokens 128 --timing`:

| Stage | CPU s | Metal s | Metal / CPU |
| --- | ---: | ---: | ---: |
| layout_vision_encode | 39.9437 | 154.6549 | 3.87x |
| layout_generate | 9.3871 | 28.4980 | 3.04x |
| content_region_vision_encode | 28.4575 | 71.3370 | 2.51x |
| content_region_generate | 10.2657 | 42.0095 | 4.09x |
| content_total | 38.8465 | 113.4659 | 2.92x |
| page_total | 92.0258 | 300.6242 | 3.27x |

Previous page 224 full-content128 Metal timing was `1385.3995s` total, with
`layout_generate + content_region_generate` accounting for about `1175.36s`.
After cache-backed decode, the same generation stages account for about
`70.51s` and page total is `300.6242s`.

Accuracy, CPU versus Metal after KV-cache decode:

| Metric | Value |
| --- | ---: |
| Block count exact | true |
| Ordered type accuracy | 1.0000 |
| Ordered mean bbox IoU | 1.0000 |
| Ordered median bbox IoU | 1.0000 |
| Mean content token F1 | 1.0000 |
| Table exact cell recall | 1.0000 |

Interpretation:

- KV-cache decode removed the dominant repeated full-prefill cost. Page 224
  full-content128 improved from `1385.40s` to `300.62s`, about `4.61x` faster.
- Metal is still slower than CPU on this benchmark, now by about `3.27x`
  instead of `13.12x`.
- The next largest remaining cost is vision encode plus per-token/per-layer
  host/device buffer churn. Persistent Metal buffers for weights, K/V cache,
  intermediate activations, and logits are now the highest-value optimization.

## Full-content512 KV-cache Checkpoint

Date: 2026-06-19
Branch: `codex/mineru-metal-backend`

After the cache-backed decode path was committed, page 224 was re-run with
`--content-max-new-tokens 512` to validate non-truncated table extraction.

Artifacts:

```text
/tmp/mu-benchmark-cpu-page224-fullcontent512-kvcache-baseline.json
/tmp/mu-benchmark-metal-page224-fullcontent512-kvcache.json
/tmp/mu-fullcontent512-kvcache-page224/cpu_page_0224.json
/tmp/mu-fullcontent512-kvcache-page224/metal_page_0224.json
/tmp/mu-fullcontent512-kvcache-page224/cpu-vs-metal.metrics.json
/tmp/mu-fullcontent512-kvcache-page224/transformers120-vs-metal.metrics.json
```

Page 224 full-content512 timing:

| Stage | CPU s | Metal s | Metal / CPU |
| --- | ---: | ---: | ---: |
| layout_vision_encode | 37.1779 | 153.2136 | 4.12x |
| layout_generate | 6.7742 | 35.3128 | 5.21x |
| content_region_vision_encode | 24.1663 | 97.4937 | 4.03x |
| content_region_generate | 20.8911 | 133.5595 | 6.39x |
| content_total | 45.2789 | 231.3019 | 5.11x |
| page_total | 93.0259 | 423.7176 | 4.55x |

End-to-end timing summary:

| Backend | Seconds | Blocks/types | CPU fallback rows |
| --- | ---: | --- | ---: |
| Transformers/MPS 120dpi reference | 52.93 | 3 table/footer/page_number | n/a |
| CPU reference | 93.13 | 3 table/footer/page_number | 0 |
| Metal no-fallback after KV-cache | 423.85 | 3 table/footer/page_number | 0 |
| Metal no-fallback before KV-cache | 3437.83 | 3 table/footer/page_number | 0 |

Speed ratios:

| Comparison | Ratio |
| --- | ---: |
| Metal after KV-cache / CPU | 4.55x slower |
| Metal after KV-cache / Transformers | 8.01x slower |
| Metal before KV-cache / Metal after KV-cache | 8.11x faster after KV-cache |

Accuracy, CPU versus Metal after KV-cache:

| Metric | Value |
| --- | ---: |
| Block count exact | true |
| Ordered type accuracy | 1.0000 |
| Ordered mean bbox IoU | 1.0000 |
| Ordered median bbox IoU | 1.0000 |
| Mean content token F1 | 1.0000 |
| Table exact cell recall | 1.0000 |

Accuracy, Transformers/MPS 120dpi versus Metal after KV-cache:

| Metric | Value |
| --- | ---: |
| Block count exact | true |
| Ordered type accuracy | 1.0000 |
| Ordered mean bbox IoU | 0.9444 |
| Ordered median bbox IoU | 0.9412 |
| Mean content token F1 | 1.0000 |
| Table exact cell recall | 1.0000 |

Interpretation:

- The cache-backed decode path preserves full-content page 224 accuracy at the
  512-token content limit.
- The old page 224 content512 Metal no-fallback run took `3437.83s`; the new
  run takes `423.85s`, an `8.11x` improvement.
- Metal remains slower than CPU and Transformers. The largest remaining
  full-content512 Metal stages are `layout_vision_encode` (`153.21s`),
  `content_region_generate` (`133.56s`), and `content_region_vision_encode`
  (`97.49s`). The next optimization should keep K/V cache and dense/norm
  intermediates in persistent Metal buffers.
