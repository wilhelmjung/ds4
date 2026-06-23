# Vision Attention Shape Tuning Plan

Date: 2026-06-23

## Context

The current Metal backend is still slower than same-run warm PyTorch/MPS on the
10-page full-content512 benchmark:

| Path | Total s | Mean s/page |
| --- | ---: | ---: |
| PyTorch/Transformers MPS measured | 450.9345 | 45.0935 |
| Current Metal | 651.6244 | 65.1624 |

The common gap is now the vision/layout encoder:

| Stage | Mean s/page |
| --- | ---: |
| `layout_vision_encode` | 33.7636 |
| `text_generate_decode` | 14.4107 |
| `content_region_vision_encode` | 14.4387 |

The new `MU_VISION_PROFILE_SPLIT=1` diagnostic path shows that the current
vision encoder is dominated by attention:

| Stage | Mean s/page | Share |
| --- | ---: | ---: |
| `vision_profile_attention` | 35.3951 | 58.1% |
| `vision_profile_norm1` | 6.8301 | 11.2% |
| `vision_profile_norm2` | 6.5893 | 10.8% |
| `vision_profile_fc1_gelu` | 3.6320 | 6.0% |
| `vision_profile_fc2` | 2.8576 | 4.7% |
| `vision_profile_qkv` | 2.7525 | 4.5% |

An initial no-flash gate proved correctness but rejected the rollback path as a
default:

| Path | Total s | Mean s/page | Output parity |
| --- | ---: | ---: | --- |
| Current flash default | 651.6244 | 65.1624 | baseline |
| `MU_VISION_ATTN_NO_FLASH=1` | 1004.3888 | 100.4389 | exact |

`MU_VISION_ATTN_NO_FLASH=1` improved some short-page layout vision timings but
severely regressed pages `244` and `247`, so the next optimization must tune
the current flash path instead of disabling it globally.

## Goal

Reduce `layout_vision_encode` on the Apple Silicon M5 test machine without
regressing the 10-page output parity gate or the table-heavy pages where the
current flash path is stable.

## Constraints

- Keep the current flash path as the default until a 10-page gate beats it.
- Keep `MU_VISION_ATTN_NO_FLASH=1` as a diagnostic/rollback switch only.
- Avoid broad rewrites. Prefer one shape gate or one small kernel variant per
  benchmark cycle.
- Use the 10-page full-content512 set as the promotion gate:
  `224,234,237,241,244,247,258,281,303,334`.
- Preserve exact output comparison against the current default:
  `mean_content_token_f1=1.0000`, exact block/type/bbox, and table cell recall
  `1.0000` where tables exist.
- Treat MPSGraph as an attention-only reference/prototype path. Do not migrate
  the full vision tower or dynamically create graphs per page/layer.

## Execution Steps

### 1. Add Flash Attention Shape Telemetry

Record the per-call inputs for `mu_vision_attn_rows_flash` under a diagnostic
flag:

```text
MU_VISION_ATTN_SHAPE_PROFILE=1
```

Capture at minimum:

- `rows`
- layer index if available at the call site
- flash tile constants / dispatch grid
- waited wall time for the attention stage when `MU_VISION_PROFILE_SPLIT=1` is
  also enabled

Verification:

```text
python3 -m unittest mineru.tests.test_mu_metal_kernel_sources
make -B mu-test mu
MU_VISION_ATTN_SHAPE_PROFILE=1 MU_VISION_PROFILE_SPLIT=1 MU_TIMING=1 \
  /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal --pages 224,258 --max-new-tokens 512 --timeout 1800 --timing \
  --out /tmp/mu-vision-flash-shape-profile-224-258.json \
  --save-output-dir /tmp/mu-vision-flash-shape-profile-224-258
```

Exit criteria:

- Shape/tile data appears for current flash calls.
- Default path remains unchanged without the diagnostic flag.

### 2. Build A Small Shape Matrix

Run the diagnostic profile on representative pages:

```text
224,244,247,258,281
```

These cover table-heavy, no-flash-regressed, and short/simple cases.

Exit criteria:

- Identify whether regressions correlate with `rows`, layer, tile count, or
  dispatch shape.
- Decide one candidate change only:
  - adjust flash tile constants,
  - add a second flash kernel variant for a narrow shape range,
  - or add a shape gate that chooses the existing no-flash path only where the
    10-page evidence supports it.

### 3. Implement One Candidate

Implement the smallest candidate from Step 2 behind an opt-in flag first:

```text
MU_VISION_ATTN_FLASH_TUNE=1
```

If the shape matrix does not identify a simple flash-tile change, implement a
bounded MPSGraph SDPA comparison lane instead:

```text
MU_VISION_ATTN_MPSGRAPH=1
```

That lane must cache graphs by stable shape, starting with `rows`, and should
replace only the vision attention segment. It is a benchmark/prototype path, not
a full vision tower migration.

Do not remove existing rollback flags.

Verification:

```text
python3 -m unittest mineru.tests.test_mu_text_timing_sources mineru.tests.test_mu_metal_kernel_sources
make -B mu-test mu
```

Then run the 2-page gate:

```text
MU_VISION_ATTN_FLASH_TUNE=1 MU_TIMING=1 \
  /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal --pages 224,258 --max-new-tokens 512 --timeout 1800 --timing \
  --out /tmp/mu-vision-flash-tune-224-258.json \
  --save-output-dir /tmp/mu-vision-flash-tune-224-258
```

Compare output against the current default artifact before continuing.

### 4. Run The 10-Page Promotion Gate

Only run this if the 2-page gate is exact and faster or clearly moves the target
stage in the right direction:

```text
MU_VISION_ATTN_FLASH_TUNE=1 MU_TIMING=1 \
  /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal --pages 224,234,237,241,244,247,258,281,303,334 \
  --max-new-tokens 512 --timeout 7200 --timing \
  --out /tmp/mu-vision-flash-tune-10page-20260623.json \
  --save-output-dir /tmp/mu-vision-flash-tune-10page-20260623
```

Promotion criteria:

- `fallback_rows=0`
- output comparison exact against current default
- total time beats `651.6244s`
- no severe page-level regression on pages `244` or `247`

### 5. Promote Or Reject

If the 10-page gate passes:

- make the tuned path default,
- keep an escape hatch,
- update `mineru/docs/mu-performance-report.md`,
- update the Gemini task and implementation plan files,
- commit code and docs.

If the 10-page gate fails:

- keep the candidate opt-in or remove it if it is not useful,
- document the rejected shape/tile hypothesis,
- choose the next single candidate from the shape matrix instead of expanding
  scope.

## References

- `mineru/mu_metal.m`: current `mu_gpu_vision_encode` and host-side attention
  dispatch.
- `mineru/metal/mu_vision.metal`: current vision attention kernels.
- `mineru/docs/mu-performance-report.md`: MPS/Metal comparison, split profile,
  no-flash 10-page gate, and MPSGraph decision.
- `mineru/docs/design-docs/mu-metal-design.md`: Metal backend design notes.
- Apple MPSGraph scaled dot product attention:
  https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/scaleddotproductattention%28query%3Akey%3Avalue%3Amask%3Ascale%3Aname%3A%29
- Apple `MPSGraphSDPADescriptor`:
  https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphsdpadescriptor
