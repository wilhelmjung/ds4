# MinerU Metal Backend Handoff

Date: 2026-06-19
Repo: `/Users/will/github/ds4`
Branch: `codex/mineru-metal-backend`

## Current Phase Summary

The current branch has a pure Metal correctness bridge for `mineru/mu.c` behind
`--backend metal --no-cpu-fallback`. CPU remains the default backend and the
precision reference.

The branch has now reached a stage-level validation point:

- 10 sampled pages complete layout-only generation in pure Metal no-fallback
  mode with zero fallback rows.
- Page 224 completes full-content extraction in pure Metal no-fallback mode
  with zero fallback rows.
- Page 224 Metal full-content output matches CPU exactly and matches the local
  Transformers/MPS 120dpi reference on table/footer/page-number content.
- Stage timing is available through `MU_TIMING=1` and the benchmark harness
  `--timing` flag.
- Performance is still much slower than CPU and Transformers/MPS, so the Metal
  path is validated as a correctness bridge, not as a production accelerator.

Latest relevant commit:

```text
1cc1e0f perf: record mu metal full content page224
```

Do not treat this as the final optimized Metal engine. The next stage should be
performance work, especially reducing Metal buffer creation and host/device
copies in the vision tower and then adding KV-cache decode.

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
- Content crop extraction and block-specific generation for page 224.
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

## Known Gaps

The current branch has not proven all desirable final-state properties:

- Full-content pure Metal validation has only been run for page 224. The
  10-page sample has layout-only parity, not full-content parity.
- The Metal path is much slower than CPU and Transformers/MPS. Page 224
  content512 is about 31.36x slower than CPU and about 64.95x slower than the
  Transformers/MPS 120dpi reference.
- The implementation is still a correctness bridge with repeated host/device
  transfers and repeated full-prefill generation.
- KV-cache decode is not the performance path yet.
- Temporary benchmark artifacts live under `/tmp` and may disappear.

## Next Recommended Work

1. Add persistent Metal buffer ownership for vision encode.
   - Cache frequently reused weights as `MTLBuffer`s.
   - Reuse intermediate activation buffers across the 32 vision blocks.
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

4. Re-run page 224 full-content after each optimization.
   - Preserve CPU exactness first.
   - Compare against Transformers 120dpi page 224.
   - Update `mineru/docs/mu-performance-report.md` after meaningful changes.

5. After page 224 performance improves, expand to the 10-page sample.
   - First layout-only parity/performance.
   - Then full-content for the table-heavy pages if runtime becomes reasonable.

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
