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

`tile variant` means the same flash attention algorithm compiled with a
different query/key block shape, for example changing the current
`query_tile_rows=32`, `key_tile_rows=32` split. It is a narrow kernel A/B for a
specific shape such as `rows=5476`, not a rewrite of the attention path.

Result, 2026-06-23:

- Implemented `MU_VISION_ATTN_SHAPE_PROFILE=1`.
- Updated `mineru/tests/mu_benchmark_pages.py` so `mu_profile` rows are
  preserved in each benchmark row's `profiles` field.
- Ran the shape matrix on `224,244,247,258,281`.
- All pages share the same layout attention shape:
  `rows=5476`, `threadgroups=172x16x1`, `query_tile_rows=32`,
  `key_tile_rows=32`, `heads=16`.
- Table-heavy pages add large content-region shapes:
  - page `224`: `3920`
  - pages `244,247`: `4144`
- Short/simple pages add small content-region shapes around `272-348`.

Decision:

- Do not add a no-flash row-shape gate. The earlier no-flash regression on
  pages `244` and `247` cannot be explained by layout row shape because layout
  rows are identical across the sampled pages.
- Target `rows=5476` first; it is the fixed layout cost on every page.
- If there is no simple flash tile variant, use the bounded
  `MU_VISION_ATTN_MPSGRAPH=1` attention-only comparison lane as the next
  prototype.

### 3. Implement One Candidate

Implement the smallest candidate from Step 2 behind an opt-in flag first:

```text
MU_VISION_ATTN_FLASH_K16=1
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
MU_VISION_ATTN_FLASH_K16=1 MU_TIMING=1 \
  /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal --pages 224,258 --max-new-tokens 512 --timeout 1800 --timing \
  --out /tmp/mu-vision-flash-k16-224-258.json \
  --save-output-dir /tmp/mu-vision-flash-k16-224-258
```

Compare output against the current default artifact before continuing.

Result, 2026-06-23:

- Added an opt-in `mu_vision_attn_rows_flash_k16` kernel and host dispatch
  switch:

  ```text
  MU_VISION_ATTN_FLASH_K16=1
  ```

- The default flash path is unchanged unless the flag is set; the K16 pipeline
  is loaded only when `MU_VISION_ATTN_FLASH_K16=1` is present.
- Verified the K16 path with `MU_VISION_ATTN_SHAPE_PROFILE=1`; profile rows
  report `path=flash_k16`, `query_tile_rows=32`, `key_tile_rows=16`.
- Source/build verification passed:
  - `python3 -m unittest mineru.tests.test_mu_benchmark_pages mineru.tests.test_mu_text_timing_sources mineru.tests.test_mu_metal_kernel_sources`
  - `make -B mu-test mu`
  - `git diff --check`

Back-to-back 2-page A/B on pages `224,258`:

| Path | Total s | Mean s/page | Mean layout vision s/page | Mean content vision s/page | Fallback rows | Output parity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default flash | 176.4189 | 88.2094 | 42.7684 | 15.4080 | 0 | baseline |
| `MU_VISION_ATTN_FLASH_K16=1` | 136.6505 | 68.3253 | 37.2810 | 11.7571 | 0 | exact |

The 2-page result justified a 10-page promotion gate, but was not sufficient to
promote the candidate by itself.

### 4. Run The 10-Page Promotion Gate

Only run this if the 2-page gate is exact and faster or clearly moves the target
stage in the right direction:

```text
MU_VISION_ATTN_FLASH_K16=1 MU_TIMING=1 \
  /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal --pages 224,234,237,241,244,247,258,281,303,334 \
  --max-new-tokens 512 --timeout 7200 --timing \
  --out /tmp/mu-vision-flash-k16-10page-20260623.json \
  --save-output-dir /tmp/mu-vision-flash-k16-10page-20260623
```

Promotion criteria:

- `fallback_rows=0`
- output comparison exact against current default
- total time beats `651.6244s`
- no severe page-level regression on pages `244` or `247`

Result, 2026-06-23:

| Path | Completed | Fallback rows | Total s | Mean s/page | Mean layout vision s/page | Mean content vision s/page |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current default baseline | 10 / 10 | 0 | 651.6244 | 65.1624 | 33.7636 | 14.4387 |
| `MU_VISION_ATTN_FLASH_K16=1` | 10 / 10 | 0 | 768.2557 | 76.8256 | 36.5858 | 15.6319 |

Page-level K16 timings:

| Page | Total s | Layout vision s | Content vision s | Decode s |
| ---: | ---: | ---: | ---: | ---: |
| 224 | 103.4306 | 40.5307 | 22.8193 | 30.8133 |
| 234 | 101.1797 | 36.5407 | 24.1772 | 36.9858 |
| 237 | 84.9311 | 34.9069 | 22.3345 | 24.4867 |
| 241 | 78.2097 | 31.6461 | 20.5536 | 23.3037 |
| 244 | 76.9071 | 30.4405 | 20.5493 | 23.3760 |
| 247 | 99.1699 | 36.1946 | 27.6420 | 31.4590 |
| 258 | 56.2694 | 39.2638 | 4.3986 | 9.8833 |
| 281 | 55.4792 | 39.7893 | 4.1067 | 9.0669 |
| 303 | 53.6441 | 36.7479 | 4.6785 | 9.2936 |
| 334 | 59.0349 | 39.7979 | 5.0594 | 10.9374 |

Decision:

- Reject K16 as a promoted/default path. It failed the 10-page total-time
  threshold and did not improve the mean layout/content vision stages versus
  the current default baseline.
- Keep `MU_VISION_ATTN_FLASH_K16=1` only as an opt-in diagnostic/tile
  comparison lane for now.
- Do not spend another cycle on blind key tile-size changes without per-kernel
  or MPSGraph evidence.

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

Current decision:

- K16 is retained as opt-in diagnostic only.
- The next single candidate is the bounded
  `MU_VISION_ATTN_MPSGRAPH=1` attention-only comparison lane. It should start
  with the stable layout shape `rows=5476`, cache the graph/pipeline by shape,
  and run the same 2-page and 10-page gates before any promotion.

MPSGraph result, 2026-06-23:

- Added an MPSGraph SDPA attention-only lane for vision attention.
- The lane packs Q/K/V to `[1,16,rows,80]`, runs
  `scaledDotProductAttentionWithQueryTensor`, reshapes back to `[rows,1280]`,
  and caches the graph by `rows`.
- The 10-page promotion gate passed, so MPSGraph vision attention is now the
  default path.
- Rollback switch:

  ```text
  MU_VISION_ATTN_NO_MPSGRAPH=1
  ```

10-page promotion gate:

| Path | Completed | Fallback rows | Total s | Mean s/page | Mean layout vision s/page | Mean content vision s/page | Output parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Previous flash default | 10 / 10 | 0 | 651.6244 | 65.1624 | 33.7636 | 14.4387 | baseline |
| MPSGraph attention default | 10 / 10 | 0 | 565.2294 | 56.5229 | 17.2824 | 9.9374 | exact |

Decision:

- Promote MPSGraph vision attention to default because it is `1.1528x` faster
  than the previous default and exact on all 10 output JSON files.
- Keep `MU_VISION_ATTN_NO_MPSGRAPH=1` as the rollback path to the previous
  flash/K16/no-flash diagnostics.
- Current Metal remains `1.2535x` slower than same-run warm PyTorch/MPS
  (`450.9345s`), so the next target is reducing MPSGraph pack/copy and command
  boundary overhead before attempting deeper MSL rewrites.

MPSGraph split profile, 2026-06-23:

- Added diagnostic-only split timing:

  ```text
  MU_VISION_ATTN_MPSGRAPH_PROFILE=1
  ```

- Full-content512 results:

| Page | Split total s | Pack QKV | MPSGraph SDPA | Copy/round | Alloc + boundary |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 4.3584 | 15.8% | 81.8% | 2.0% | 0.4% |
| 258 | 2.9407 | 11.9% | 84.0% | 3.5% | 0.6% |

- Skip-content page `258` showed the same shape: MPSGraph SDPA was `88.7%` of
  the split.
- Page `224` and `258` profile outputs were byte-level exact against the
  current MPSGraph default artifacts.

Updated decision:

- Buffer allocation and copy/round are not the next useful target.
- Pack QKV is visible but still secondary.
- The dominant remaining local M5 cost is the MPSGraph SDPA call itself.
- Per the latest user instruction, ignore GB10 for this optimization choice.
- Next candidate: one opt-in custom MSL SDPA/attention lane for stable layout
  `rows=5476`, using MPSGraph as the correctness oracle and keeping MPSGraph as
  default until the 2-page and 10-page gates prove a faster exact path.

Packed MSL 5476 result, 2026-06-23:

- Added an opt-in packed MSL comparison lane:

  ```text
  MU_VISION_ATTN_MSL_PACKED_5476=1
  ```

- The lane applies only when `rows == 5476`; other shapes continue using the
  default MPSGraph path.
- It reuses the existing MPSGraph Q/K/V pack layout and then runs
  `mu_vision_attn_rows_packed_flash`.
- Page `258` skip-content, same `max_new_tokens=512` gate:

| Path | Total s | Layout vision s | Fallback rows | Output parity |
| --- | ---: | ---: | ---: | --- |
| MPSGraph default | 30.6620 | 18.9131 | 0 | baseline |
| `MU_VISION_ATTN_MSL_PACKED_5476=1` | 46.5352 | 34.0411 | 0 | byte-level exact |

Decision:

- Reject packed MSL 5476 as a promotion candidate. It is exact, but already
  `1.52x` slower in total page time and `1.80x` slower in layout vision on the
  fastest realistic gate.
- Do not run the longer 2-page or 10-page gates for this lane.
- Keep it as an opt-in diagnostic comparison lane only.
- The next local M5 candidate needs a materially different attention design
  such as simdgroup/tiled SDPA, or it should target a non-attention stage.

Post-MPSGraph non-attention profile, 2026-06-24:

- Ran page `258` skip-content with both split profilers enabled:

  ```text
  MU_VISION_PROFILE_SPLIT=1 MU_VISION_ATTN_MPSGRAPH_PROFILE=1 MU_TIMING=1
  ```

- Result:

| Stage | Seconds | Share of layout vision |
| --- | ---: | ---: |
| `vision_profile_norm1` | 3.5621 | 21.5% |
| `vision_profile_norm2` | 3.5172 | 21.2% |
| `vision_attn_mpsgraph_graph` | 2.0094 | 12.1% |
| `vision_profile_fc1_gelu` | 1.6800 | 10.1% |
| `vision_profile_fc2` | 1.4098 | 8.5% |
| `vision_profile_qkv` | 1.2396 | 7.5% |

- Output stayed exact against the same-day default MPSGraph profile output.

Updated decision:

- Do not start another attention variant next.
- Next shortest useful implementation target is vision LayerNorm, because
  `norm1 + norm2` is the largest remaining local M5 cost.
- If LayerNorm has no cheap win, move to the FFN pair (`fc1_gelu + fc2`).

LayerNorm SIMD result, 2026-06-24:

- Added `mu_layernorm_bf16_rows_simd` for vision `cols == 1280`.
- Promoted it to default.
- Rollback:

  ```text
  MU_VISION_LAYERNORM_NO_SIMD=1
  ```

Page `258` skip-content split profile:

| Path | Page total s | Layout vision s | Norm1 s | Norm2 s |
| --- | ---: | ---: | ---: | ---: |
| Previous default | 26.5270 | 16.5731 | 3.5621 | 3.5172 |
| SIMD LayerNorm default | 17.1204 | 7.6873 | 0.0428 | 0.0263 |

Two-page full-content gate:

| Path | Total s | Mean s/page | Mean layout vision s/page | Mean content vision s/page | Output parity |
| --- | ---: | ---: | ---: | ---: | --- |
| Previous default | 72.2379 | 36.1189 | 12.3955 | 5.8953 | baseline |
| SIMD LayerNorm default | 57.7566 | 28.8783 | 7.2973 | 3.4085 | exact |

Decision:

- Keep SIMD LayerNorm as default.
- Do not spend the next cycle on LayerNorm.
- Before committing to the vision FFN pair, rerun the 10-page MPS comparison to
  verify the current E2E bottleneck after LayerNorm promotion.

Fresh MPS comparison after SIMD LayerNorm, 2026-06-24:

- Ran PyTorch/MPS warm-up, then PyTorch/MPS measured on the same 10-page set
  with `batch-size=1`, `dpi=120`, `max_new_tokens=512`.
- Rebuilt `mu`, then ran current Metal with `MU_TIMING=1`,
  `--backend metal`, and `--no-cpu-fallback`.
- Artifacts:

  ```text
  /tmp/mu-mps-b2b-10page-warm-20260624/summary.json
  /tmp/mu-mps-b2b-10page-measured-20260624/summary.json
  /tmp/mu-mps-b2b-10page-measured-20260624/pages.jsonl
  /tmp/mu-metal-layernorm-simd-b2b-10page-20260624.json
  /tmp/mu-metal-layernorm-simd-b2b-10page-20260624/metal_page_*.json
  /tmp/mu-metal-vs-mps-b2b-10page-20260624.metrics.json
  ```

| Path | Completed | Total s | Mean s/page | Relative time |
| --- | ---: | ---: | ---: | ---: |
| PyTorch/MPS measured | `10 / 10` | `521.6959` | `52.1696` | `1.0000x` |
| Current Metal | `10 / 10` | `328.0251` | `32.8025` | `0.6288x` |

- Current Metal has `0` fallback rows and is `1.5904x` faster than fresh
  PyTorch/MPS on this local M5 gate.
- Output comparison against PyTorch/MPS measured output stayed exact for block
  count, type order, content token F1, and table cells; mean bbox IoU was
  `0.9877`.
- Mean Metal stage timings:
  - `text_generate_decode=18.2951s/page`
  - `content_region_generate=15.2968s/page`
  - `layout_vision_encode=7.4304s/page`
  - `content_region_vision_encode=3.9401s/page`

Updated decision:

- Current native Metal is no longer slower than the local PyTorch/MPS baseline
  on the 10-page gate.
- Use MPS as a regression reference, not as the immediate blocker.
- Next local M5 target is per-kernel timing and reduction of
  `text_generate_decode` / `content_region_generate`.
- The remaining vision FFN pair stays as the secondary target if decoder
  dispatch overhead is not cheaply reducible.

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
