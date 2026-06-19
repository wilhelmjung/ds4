# MinerU Metal Backend Handoff

Date: 2026-06-19
Repo: `/Users/will/github/ds4`
Branch: `codex/mineru-metal-backend`

## Current Phase Summary

The current branch has a pure Metal correctness bridge for `mineru/mu.c` behind
`--backend metal --no-cpu-fallback`. CPU remains the default backend and the
precision reference.

The branch has now reached a 10-page full-content validation point:

- 10 sampled pages complete layout-only generation in pure Metal no-fallback
  mode with zero fallback rows.
- 10 sampled pages complete full-content512 extraction in pure Metal
  no-fallback mode with zero fallback rows.
- The 10-page Metal full-content512 output matches CPU exactly: block counts,
  ordered block types, bbox IoU, content token F1, and all table cells.
- The same outputs also match the local Transformers/MPS 120dpi reference on
  block counts, ordered block types, content token F1, and all table cells in
  the smoke corpus.
- Stage timing is available through `MU_TIMING=1` and the benchmark harness
  `--timing` flag.
- Performance is still much slower than CPU and Transformers/MPS, so the Metal
  path is validated as a correctness bridge, not as a production accelerator.

Latest relevant commit:

```text
f6bd048 perf: record mu metal kv-cache content512
```

Do not treat this as the final optimized Metal engine. The next stage should be
performance work, especially persistent Metal buffers for weights,
intermediates, K/V cache, and logits to reduce buffer creation and host/device
copies.

## Worktree State

Expected current `git status --short --branch` shape:

```text
## codex/mineru-metal-backend
?? download_model.q2-imatrix.log
?? mu
?? mu-test
```

These untracked files are local artifacts and should not be committed:

- `download_model.q2-imatrix.log`
- `mu`
- `mu-test`

This handoff file itself is newly added and may be committed separately if the
next operator wants a persistent handoff artifact.

## Implemented Metal Coverage

The Metal path currently covers enough of the MinerU page pipeline to run:

- Qwen2-VL image preprocessing and vision encode through the native Metal path.
- Layout generation through `--backend metal --no-cpu-fallback`.
- Content crop extraction and block-specific generation for the 10-page
  full-content512 smoke corpus.
- JSON output for parsed layout/content blocks.
- CPU fallback accounting and benchmark rejection of fallback via
  `--no-cpu-fallback`.

Relevant source areas:

- `mineru/mu.c`
- `mineru/mu_cli.c`
- `mineru/mu_gpu.h`
- `mineru/mu_metal.m`
- `mineru/metal/mu_dense.metal`
- `mineru/metal/mu_norm.metal`
- `mineru/metal/mu_vision.metal`
- `mineru/metal/mu_attn.metal`
- `mineru/tests/mu_benchmark_pages.py`
- `mineru/tests/mu_compare_outputs.py`

## Key Validation Evidence

Detailed numbers are in:

```text
mineru/docs/mu-performance-report.md
```

### 10-page Full-content512 E2E

Artifacts:

```text
/tmp/mu-benchmark-metal-page224-fullcontent512-kvcache.json
/tmp/mu-benchmark-metal-9remaining-fullcontent512-kvcache.json
/tmp/mu-benchmark-cpu-page224-fullcontent512-kvcache-baseline.json
/tmp/mu-benchmark-cpu-9remaining-fullcontent512-kvcache-baseline.json
/tmp/mu-fullcontent512-kvcache-10-combined/cpu-vs-metal.metrics.json
/tmp/mu-fullcontent512-kvcache-10-combined/transformers120-vs-metal.metrics.json
```

Result:

| Metric | Value |
| --- | ---: |
| Pages | 10 |
| Metal completed pages | 10 / 10 |
| Metal fallback rows | 0 |
| CPU-vs-Metal exact block-count pages | 10 / 10 |
| CPU-vs-Metal ordered type accuracy | 1.0000 |
| CPU-vs-Metal ordered mean bbox IoU | 1.0000 |
| CPU-vs-Metal mean content token F1 | 1.0000 |
| CPU-vs-Metal table exact cells | 104 / 104 |
| Transformers-vs-Metal ordered mean bbox IoU | 0.9877 |
| Mean content token F1 | 1.0000 |
| Table exact cells | 104 / 104 |
| Table exact cell recall | 1.0000 |
| CPU total time | 930.73s |
| CPU mean time | 93.07s/page |
| Metal total time | 2720.83s |
| Metal mean time | 272.08s/page |
| Transformers/MPS total time | 385.77s |
| Transformers/MPS mean time | 38.58s/page |
| Metal / CPU speed gap | 2.92x slower |
| Metal / Transformers speed gap | 7.05x slower |

Pages:

```text
224, 234, 237, 241, 244, 247, 258, 281, 303, 334
```

This is the current strongest end-to-end validation evidence. CPU remains the
precision reference and is now proven exact against Metal for the 10-page
full-content512 sample. Do not rerun this full CPU baseline for every Metal-only
optimization; reuse it unless CPU code, parsing semantics, token limits, model
weights, or comparison logic change.

### 10-page Layout-only Parity

Command shape:

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --max-new-tokens 128 \
  --skip-content \
  --resume \
  --out /tmp/mu-benchmark-metal-layout128-batch*.json
```

Result recorded in the performance report:

| Backend | Total s | Mean s/page | Completed pages | Fallback rows |
| --- | ---: | ---: | ---: | ---: |
| CPU reference | 511.64 | 51.16 | 10 / 10 | 0 |
| Metal no-fallback | 7435.68 | 743.57 | 10 / 10 | 0 |

The sampled pages were:

```text
224, 234, 237, 241, 244, 247, 258, 281, 303, 334
```

Metal matched CPU block count and ordered block types for all 10 pages.

### Page 224 Full-content E2E

Metal command:

```bash
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

CPU reference command:

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend cpu \
  --pages 224 \
  --max-new-tokens 128 \
  --content-max-new-tokens 512 \
  --timeout 2400 \
  --resume \
  --out /tmp/mu-benchmark-cpu-page224-fullcontent-content512.json \
  --save-output-dir /tmp/mu-fullcontent-page224-content512
```

Results:

| Backend | Seconds | Blocks/types | Fallback rows |
| --- | ---: | --- | ---: |
| Transformers/MPS 120dpi reference | 52.93 | 3 table/footer/page_number | n/a |
| CPU reference | 109.63 | 3 table/footer/page_number | 0 |
| Metal no-fallback | 3437.83 | 3 table/footer/page_number | 0 |

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

Important note: the earlier full-content diagnostic with only
`--max-new-tokens 128` and no `--content-max-new-tokens` proves CPU/Metal exact
parity but truncates table content. Use `--content-max-new-tokens 512` when
comparing content completeness to Transformers.

## Temporary Artifacts

Useful artifacts from the current validation stage:

```text
/tmp/mu-benchmark-cpu-page224-fullcontent-content512.json
/tmp/mu-benchmark-metal-page224-fullcontent-content512.json
/tmp/mu-fullcontent-page224-content512/cpu_page_0224.json
/tmp/mu-fullcontent-page224-content512/metal_page_0224.json
/tmp/mu-fullcontent-page224-content512/cpu-vs-metal.metrics.json
/tmp/mu-fullcontent-page224-content512/transformers120-vs-cpu.metrics.json
/tmp/mu-fullcontent-page224-content512/transformers120-vs-metal.metrics.json
/tmp/mu-benchmark-cpu-page224-token1-timing.json
/tmp/mu-benchmark-metal-page224-token1-timing.json
/tmp/mu-benchmark-cpu-page224-fullcontent128-timing.json
/tmp/mu-benchmark-metal-page224-fullcontent128-timing.json
/tmp/mu-fullcontent128-timing-page224/cpu_page_0224.json
/tmp/mu-fullcontent128-timing-page224/metal_page_0224.json
/tmp/mu-fullcontent128-timing-page224/cpu-vs-metal.metrics.json
```

The Transformers reference used for page 224:

```text
/Users/will/github/mineru-model/runs/nasa_systems_engineering_handbook_rev2_page224_mps_120dpi_test/pages.jsonl
```

These `/tmp` files are useful for immediate inspection but should be considered
ephemeral. The stable record is the committed performance report.

## Verification Commands Already Run

Most recent verification for the current stage:

```bash
/Users/will/github/mineru-model/.venv/bin/python -m unittest \
  mineru.tests.test_mu_benchmark_pages \
  mineru.tests.test_mu_compare_outputs

make mu-test

git diff --check

./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json

/Users/will/github/mineru-model/.venv/bin/python -m json.tool \
  /tmp/mu-benchmark-cpu-page224-fullcontent-content512.json
/Users/will/github/mineru-model/.venv/bin/python -m json.tool \
  /tmp/mu-benchmark-metal-page224-fullcontent-content512.json
/Users/will/github/mineru-model/.venv/bin/python -m json.tool \
  /tmp/mu-fullcontent-page224-content512/cpu-vs-metal.metrics.json
/Users/will/github/mineru-model/.venv/bin/python -m json.tool \
  /tmp/mu-fullcontent-page224-content512/transformers120-vs-cpu.metrics.json
/Users/will/github/mineru-model/.venv/bin/python -m json.tool \
  /tmp/mu-fullcontent-page224-content512/transformers120-vs-metal.metrics.json
```

Expected outcomes:

- Python unit tests: 5 tests pass.
- `make mu-test`: prints `mu_test ok`.
- `git diff --check`: no output.
- CPU and Metal trace commands: exit 0 and print the trace `ok` lines.
- JSON artifacts: parse cleanly.

## Stage Timing Evidence

The latest timing smoke used page 224 with `--max-new-tokens 1 --skip-content`
to isolate fixed per-page costs:

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --pages 224 \
  --max-new-tokens 1 \
  --skip-content \
  --timing \
  --timeout 1200 \
  --out /tmp/mu-benchmark-metal-page224-token1-timing.json
```

Key numbers:

| Stage | CPU s | Metal s | Metal / CPU |
| --- | ---: | ---: | ---: |
| layout_vision_encode | 39.5267 | 101.0340 | 2.56x |
| layout_generate | 4.6844 | 7.6832 | 1.64x |
| layout_prompt_tokenize | 3.7702 | 3.8659 | 1.03x |
| page_total | 48.0315 | 112.6486 | 2.35x |

This is the strongest current evidence for the next optimization target:
full-page vision encode dominates the fixed per-page cost. Optimize Metal buffer
reuse and host/device transfers there before spending time on broad
full-content sampling.

A second timing run used page 224 full-content mode with layout
`--max-new-tokens 128` and `--content-max-new-tokens 128`. This is not the
content-completeness reference, because 128 content tokens truncate the table,
but CPU and Metal remain exactly equal and it exercises the full three-block
content loop.

Key full-content128 numbers:

| Stage | CPU s | Metal s | Metal / CPU |
| --- | ---: | ---: | ---: |
| layout_vision_encode | 49.1435 | 135.5313 | 2.76x |
| layout_generate | 8.2231 | 494.9236 | 60.19x |
| content_region_vision_encode | 32.2100 | 70.3365 | 2.18x |
| content_region_generate | 11.9075 | 680.4337 | 57.14x |
| content_total | 44.2390 | 750.9101 | 16.97x |
| page_total | 105.5664 | 1385.3995 | 13.12x |

Revised performance conclusion:

- For fixed per-page cost, `layout_vision_encode` is still the main front-half
  bottleneck.
- For full-content extraction, repeated full-prefill generation is now the
  largest measured Metal cost. `layout_generate + content_region_generate` is
  about `1175.36s` out of `1385.40s` total.
- The next optimization path should include both persistent Metal buffers for
  vision encode and KV-cache decode or an equivalent strategy to avoid repeated
  full-prefill generation.

## KV-cache Decode Checkpoint

Date: 2026-06-19

The Metal generation path now has a cache-backed decode implementation. Prefill
still uses the existing Metal sequence kernels, but it records per-layer K/V
cache rows. Decode tokens now use one-token Metal dense/norm/MLP kernels plus
`mu_text_attn_cached`.

Validation completed:

```bash
make mu-test
make mu
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_metal_text_generation_smoke.py
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_metal_layout_generation_smoke.py
```

The Metal generation smoke tests now assert `mu metal stage: text_cached_attn`
so they catch regressions back to repeated full-prefill generation.

Page 224 layout-only timing, `--max-new-tokens 128 --skip-content --timing`:

| Backend | Seconds | layout_generate s | layout_vision_encode s |
| --- | ---: | ---: | ---: |
| CPU | 54.35 | 8.83 | 40.99 |
| Metal no-fallback | 147.18 | 29.10 | 113.69 |

Page 224 full-content128 timing:

| Backend | Seconds | layout_generate s | content_region_generate s |
| --- | ---: | ---: | ---: |
| CPU | 92.11 | 9.39 | 10.27 |
| Metal no-fallback | 300.76 | 28.50 | 42.01 |

CPU versus Metal output metrics for
`/tmp/mu-fullcontent128-kvcache-page224` remain exact:

| Metric | Value |
| --- | ---: |
| Block count exact | true |
| Ordered type accuracy | 1.0000 |
| Ordered mean bbox IoU | 1.0000 |
| Mean content token F1 | 1.0000 |
| Table exact cell recall | 1.0000 |

Key artifacts:

```text
/tmp/mu-benchmark-metal-page224-layout128-kvcache.json
/tmp/mu-benchmark-cpu-page224-layout128-kvcache-baseline.json
/tmp/mu-benchmark-metal-page224-fullcontent128-kvcache.json
/tmp/mu-benchmark-cpu-page224-fullcontent128-kvcache-baseline.json
/tmp/mu-fullcontent128-kvcache-page224/cpu-vs-metal.metrics.json
```

Updated performance conclusion:

- KV-cache decode reduced page 224 full-content128 from `1385.40s` to
  `300.62s`, about `4.61x` faster than the previous Metal timing.
- Metal is still about `3.27x` slower than CPU for page 224 full-content128.
- The next highest-value work is persistent Metal buffer ownership for weights,
  K/V cache, intermediate activations, and logits to remove per-kernel
  `newBufferWithBytes` and host/device copies.

## Full-content512 KV-cache Validation

Date: 2026-06-19

Page 224 was re-run with `--content-max-new-tokens 512` after commit
`46d7841`.

Artifacts:

```text
/tmp/mu-benchmark-metal-page224-fullcontent512-kvcache.json
/tmp/mu-benchmark-cpu-page224-fullcontent512-kvcache-baseline.json
/tmp/mu-fullcontent512-kvcache-page224/cpu_page_0224.json
/tmp/mu-fullcontent512-kvcache-page224/metal_page_0224.json
/tmp/mu-fullcontent512-kvcache-page224/cpu-vs-metal.metrics.json
/tmp/mu-fullcontent512-kvcache-page224/transformers120-vs-metal.metrics.json
```

Timing:

| Backend | Seconds | Blocks/types | Fallback rows |
| --- | ---: | --- | ---: |
| Transformers/MPS 120dpi reference | 52.93 | 3 table/footer/page_number | n/a |
| CPU reference | 93.13 | 3 table/footer/page_number | 0 |
| Metal no-fallback after KV-cache | 423.85 | 3 table/footer/page_number | 0 |
| Metal no-fallback before KV-cache | 3437.83 | 3 table/footer/page_number | 0 |

Stage highlights:

| Stage | CPU s | Metal s | Metal / CPU |
| --- | ---: | ---: | ---: |
| layout_vision_encode | 37.1779 | 153.2136 | 4.12x |
| layout_generate | 6.7742 | 35.3128 | 5.21x |
| content_region_vision_encode | 24.1663 | 97.4937 | 4.03x |
| content_region_generate | 20.8911 | 133.5595 | 6.39x |
| content_total | 45.2789 | 231.3019 | 5.11x |
| page_total | 93.0259 | 423.7176 | 4.55x |

Accuracy:

| Comparison | Content F1 | Table exact cell recall | BBox mean IoU |
| --- | ---: | ---: | ---: |
| CPU vs Metal | 1.0000 | 1.0000 | 1.0000 |
| Transformers/MPS 120dpi vs Metal | 1.0000 | 1.0000 | 0.9444 |

The full-content512 Metal no-fallback path is now validated for page 224 with
complete table content. It is about `8.11x` faster than the pre-KV-cache
content512 Metal run, but still about `4.55x` slower than CPU and `8.01x`
slower than Transformers/MPS.

## Known Gaps

The current branch has not proven all desirable final-state properties:

- Metal is still slower than CPU and Transformers/MPS. Page 224
  full-content512 is about 4.55x slower than CPU. The 10-page full-content512
  run is about 2.92x slower than CPU and 7.05x slower than Transformers/MPS.
- The implementation still has repeated host/device transfers and does not own
  persistent Metal buffers for weights, intermediate activations, K/V cache, or
  logits.
- Temporary benchmark artifacts live under `/tmp` and may disappear.

## Next Recommended Work

1. Add persistent Metal buffer ownership.
   - Cache frequently reused weights as `MTLBuffer`s.
   - Reuse intermediate activation buffers across text and vision kernels.
   - Keep K/V cache in Metal buffers during decode instead of copying slices
     through shared host memory.
   - Avoid repeated `newBufferWithBytes` and CPU-side `memcpy` per dense/norm
     kernel.

2. Profile and reduce full-page vision encode cost.
   - Start with page 224 because the current report has CPU, Metal, and
     Transformers reference numbers.
   - Keep `--backend metal --no-cpu-fallback` as the only benchmark mode that
     counts.

3. Add or improve timing instrumentation.
   - Split page time into image preprocess, vision encode, layout generation,
     crop extraction, and content generation.
   - Record host/device bytes moved if practical.

4. Re-run page 224 full-content512 after each optimization.
   - Preserve CPU exactness first.
   - Compare against Transformers 120dpi page 224.
   - Update `mineru/docs/mu-performance-report.md` after meaningful changes.

5. After page 224 performance improves further, re-run the 10-page
   full-content512 sample.
   - Require `--backend metal --no-cpu-fallback`.
   - Compare against the existing Transformers/MPS reference.
   - Reuse the existing CPU full-content512 baseline unless CPU code, parsing
     semantics, token limits, model weights, or comparison logic changed.

## Safety Rules For Next Operator

- Keep CPU as the default backend and precision reference.
- Do not treat fallback-enabled Metal timing as Metal performance.
- Do not commit generated binaries or transient logs.
- Use `apply_patch` for manual source/doc edits.
- Run CPU gates before Metal gates after C, Objective-C, Metal, or Makefile
  changes.
- Do not delete or overwrite user changes if the worktree becomes dirty.

## Quick Commands

Build:

```bash
make -B mu
make mu-test
```

Trace gates:

```bash
./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
```

Page 224 full-content Metal benchmark:

```bash
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

Compare outputs:

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_compare_outputs.py \
  --ref-json /tmp/mu-fullcontent-page224-content512/cpu_page_0224.json \
  --pred-json /tmp/mu-fullcontent-page224-content512/metal_page_0224.json \
  --out /tmp/mu-fullcontent-page224-content512/cpu-vs-metal.metrics.json
```
