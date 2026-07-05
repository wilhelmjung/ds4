# Handoff Report: Metal Backend Optimizations & Performance Baseline

## 1. Accomplished Work (Phases 23-25)

### Indirect Command Buffer (ICB) Concurrency Fix
- **Problem**: Concurrent layout extraction (e.g. `--threads 2` or `4` workers) with `MU_TEXT_DECODE_ICB=1` caused data races because the pre-recorded ICB bound absolute, zero-based scratch buffer offsets, leading threads to overwrite intermediate hidden states and output logits.
- **Solution**: Refactored `mu_gpu_text_decode_icb_record` and `mu_gpu_text_decode_icb_execute` in [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m) to calculate thread-local base offsets: `base_offset = tl_worker_id * chunk`. Both host/device transfers and kernel execution now partition scratch memory safely.
- **Result**: Concurrent execution is fully thread-safe, returning 100% correct, bit-exact layouts matching the CPU reference baseline.

### CLI Optimization Flags
- Integrated `--kv-cache-bf16` and `--use-icb` in [mu_cli.c](file:///Users/will/github/ds4/mineru/mu_cli.c) to set their respective environment variables programmatically, removing the need for manual configuration.
- Supported forwarding the new switches in [mu_benchmark_pages.py](file:///Users/will/github/ds4/mineru/tests/mu_benchmark_pages.py).

### Automated Regression Testing
- Created a self-contained regression check [mu_regress_check.py](file:///Users/will/github/ds4/mineru/tests/mu_regress_check.py) that renders reference page 224, runs `./mu` in single-threaded and concurrent modes with the optimizations active, and asserts the output layout is correct.
- Defined a `mu-regress` target in the [Makefile](file:///Users/will/github/ds4/Makefile) and linked it to the Darwin `mu-test` target, run automatically via `make mu-test`.

### Test Suite VRAM Memory Leak & Cache Fixes
- Wrapped `mu_gpu_destroy` in `@autoreleasepool` in [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m) to force immediate reclaim of Metal buffers, preventing process OOM crashes (`Killed: 9`).
- Refactored [mu_test.c](file:///Users/will/github/ds4/mineru/tests/mu_test.c) to reuse a single `global_gpu` context and declared weight arrays as `static const` to prevent cache address collisions.

### Decoder ICB Profiling & QKV/RoPE Fusion Fix
- Added `text_generate_decode_cached_icb` timing so ICB decode time is visible instead of hidden behind zeroed split buckets.
- Made `MU_TEXT_DECODE_PROFILE_SPLIT=1` bypass `MU_TEXT_DECODE_ICB=1`, keeping split profiling diagnostic and comparable.
- Fixed and promoted the ICB-recorded QKV/RoPE fusion: dynamic parameters now bind in shader order (`pos3`, `cache_pos`, `cols`, `use_bf16_cache`), and the fused dispatch now covers the actual 640 output rows instead of 1152.
- The ICB path now fuses QKV projection and RoPE/KV-cache update by default. `MU_TEXT_DECODE_QKV_ROPE_NO_FUSION=1` keeps the old two-command ICB path available for comparison.
- Fresh 5-run adjacent layout trace A/B with `--kv-cache-bf16 --use-icb`: fused QKV/RoPE averaged `0.1808s`; old ICB QKV/RoPE path averaged `0.1974s`. Post-promotion 3-run A/B was closer (`0.1946s` vs `0.1967s`) but kept the expected dispatch reduction (`585/144` vs `657/216`). All runs had trace parity OK.
- Build warning cleanup: removed an unused `name` variable in the batched decode path.

### Decoder FFN SIMDGroup Default
- Replaced the non-ICB default decoder FFN path with the existing prefill SIMDGroup SwiGLU helper plus `dense_f32_rows` down projection and residual add.
- Kept the old monolithic `mu_text_decode_fused_ffn` path behind `MU_TEXT_DECODE_FFN_NO_SIMDGROUP=1`; `MU_TEXT_NO_FUSED_FFN=1` still forces the fully unfused fallback.
- Fresh `layout` trace A/B with `--kv-cache-bf16`:
  - New default: decode `0.1091s`, trace parity OK.
  - Old path via `MU_TEXT_DECODE_FFN_NO_SIMDGROUP=1`: decode `0.1459s`, trace parity OK.
  - ICB now records RMSNorm + SIMDGroup SwiGLU + shader down projection + residual add by default. `MU_TEXT_DECODE_ICB_FFN_NO_SIMDGROUP=1` keeps the old monolithic FFN command available for comparison.
  - Fresh 3-run adjacent layout trace A/B with `--kv-cache-bf16 --use-icb`: default ICB FFN averaged `0.2088s`; old FFN escape hatch averaged `0.2211s`. Both paths had trace parity OK.

### Cached Attention Micro-Experiments
- Tried a no-score recompute variant for cached decode attention. Correctness held, but layout ICB decode regressed (`0.4319s` vs adjacent default `0.2134s`), so it was not kept.
- Tried a smaller `scores[2048]` variant for layout-sized prompts. Trace parity held, but follow-up A/B was noisy and did not produce a stable win, so it was not promoted or kept.

### Vision Split Profiling, Attention Rejection, and Dense 2SG Default
- Ran sequential `MU_VISION_PROFILE_SPLIT=1` profiling on pages 224, 244, and 303. The stable hot stages were Vision attention (`5.591659s` total across 96 blocks), FFN `fc1_gelu` (`3.325887s`), FFN `fc2` (`3.555260s`), QKV (`2.524866s`), and projection (`0.857976s`).
- Rejected existing attention variants after direct A/B:
  - `MU_VISION_ATTN_FLASH_K16=1` on page 244: `vision_encode=26.223042s`, `attention=21.264510s`.
  - `MU_VISION_ATTN_MSL_PACKED_5476=1` on page 244: `vision_encode=20.139595s`, `attention=15.439565s`.
  - Adjacent default on page 244: `vision_encode=7.119092s`, `attention=1.860926s`.
- Added two 2-simdgroup dense kernels in [mu_dense.metal](file:///Users/will/github/ds4/mineru/metal/mu_dense.metal): generic BF16+bias rows and BF16+bias+QuickGELU rows. Each threadgroup now computes two 8-row tiles for the same 32 output columns, sharing one weight tile load across two simdgroups.
- Promoted the 2SG dense path to default for existing SIMDGroup dense shapes in [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m). `MU_DENSE_ROWS_NO_2SG=1` keeps the old 1SG path available for comparison.
- 3-page default-vs-2SG profile A/B:
  - `vision_encode` mean: `7.225235s -> 6.889842s`.
  - `page_total` mean: `12.118075s -> 11.898402s`.
  - `fc1_gelu` total: `3.325887s -> 3.050101s`.
  - `fc2` total: `3.555260s -> 3.026327s`.
  - `proj` total: `0.857976s -> 0.786274s`.
- Verification:
  - `make -B mu`
  - `./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json`
  - `/Users/will/github/mineru-model/.venv/bin/python -m unittest mineru.tests.test_mu_metal_kernel_sources`
  - `git diff --check`

### Vision QKV 2SG Default
- Added `mu_dense_bf16_bias_rows_simdgroup_qkv_2sg`, applying the same two-simdgroup weight-sharing pattern to the fused Vision QKV projection.
- Terminology: `QKV` means the fused Query/Key/Value projection in each Vision attention block. `2SG` means one Metal threadgroup uses two simdgroups, computing two 8-row tiles at once.
- The 2SG QKV kernel computes 16 rows per threadgroup and lets both simdgroups share the same loaded weight tile, reducing duplicate weight reads versus the old one-simdgroup QKV path.
- Probed it behind `MU_DENSE_QKV_2SG=1`, then promoted it to default after adjacent page 224/244/303 split profiles showed stable QKV wins:
  - Page 224: `0.802685s -> 0.759387s`.
  - Page 244: `0.795874s -> 0.731034s`.
  - Page 303: `0.795189s -> 0.726481s`.
- `MU_DENSE_ROWS_NO_2SG=1` remains the escape hatch for the old one-simdgroup QKV path.
- Verification:
  - `/Users/will/github/mineru-model/.venv/bin/python -m unittest mineru.tests.test_mu_metal_kernel_sources`
  - `make -B mu`
  - `MU_METAL_DEBUG=1 ./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json`

---

## 2. Verified Performance Baseline (10-Page Timings)

Below is the refreshed sequential 10-page timing for the fully optimized Metal engine with `--kv-cache-bf16 --use-icb` active after the Vision 2SG dense and QKV 2SG defaults. The PyTorch MPS column is the existing warm-rerun reference; concurrent Metal timings were not refreshed and are intentionally omitted until worker contention is re-profiled.

Command:

```bash
/Users/will/github/mineru-model/.venv/bin/python -m mineru.tests.mu_benchmark_pages \
  --backend metal --threads 1 --pages 224,234,237,241,244,247,258,281,303,334 \
  --timeout 7200 --timing --kv-cache-bf16 --use-icb \
  --out /tmp/mu_10page_seq_qkv2sg_refresh.json \
  --save-output-dir /tmp/mu_10page_seq_qkv2sg_refresh_outputs
```

| Page | PyTorch MPS (Warm Rerun) (s) | Metal Seq (Refreshed) (s) | Speedup (Seq vs MPS) | Prior Metal Seq (s) | Seq Refresh Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Page 224 | 118.81s | 19.56s | 6.08x | 24.27s | 1.24x |
| Page 234 | 157.22s | 15.58s | 10.09x | 32.94s | 2.11x |
| Page 237 | 169.46s | 24.09s | 7.03x | 29.10s | 1.21x |
| Page 241 | 75.94s | 25.56s | 2.97x | 25.42s | 0.99x |
| Page 244 | 65.08s | 16.30s | 3.99x | 22.28s | 1.37x |
| Page 247 | 51.55s | 15.94s | 3.23x | 30.37s | 1.91x |
| Page 258 | 25.60s | 13.33s | 1.92x | 15.82s | 1.19x |
| Page 281 | 17.88s | 16.50s | 1.08x | 14.50s | 0.88x |
| Page 303 | 15.18s | 15.57s | 0.98x | 16.49s | 1.06x |
| Page 334 | 15.62s | 12.56s | 1.24x | 15.33s | 1.22x |
| **Total** | **712.34s** | **174.98s** | **4.07x** | **226.52s** | **1.29x** |
| **Mean** | **71.23s** | **17.50s** | **4.07x** | **22.65s** | **1.29x** |

- **Sequential acceleration**: refreshed Metal sequential is **4.07x faster** than the existing PyTorch MPS warm-rerun reference.
- **Refresh delta**: refreshed Metal sequential is **1.29x faster** than the prior Metal sequential baseline (`226.52s -> 174.98s` total).
- **Run status**: completed `10/10` pages, `0` failures, `0` Metal fallback rows.
- **Mean stage timings**: `page_total=17.498420s`, `layout_vision_encode=6.162498s`, `vision_encode=9.659590s`, `layout_generate=7.760307s`, `text_generate_prefill=4.605782s`, `text_generate_decode=3.103447s`, `content_total=3.538578s`.
- **Output artifacts**: `/tmp/mu_10page_seq_qkv2sg_refresh.json` and `/tmp/mu_10page_seq_qkv2sg_refresh_outputs/metal_page_*.json`.

---

## 3. Next Steps & Future Plans

1. **Push/Publish Local Commits**:
   - **Handoff Task**: Push the local branch `codex/mineru-metal-backend` after this handoff/update commit if remote publication is desired.
