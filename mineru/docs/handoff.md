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

---

## 2. Verified Performance Baseline (10-Page Timings)

Below is the comparative performance timings of the fully optimized Metal engine (with both `--kv-cache-bf16` and `--use-icb` active) compared to the **PyTorch MPS (Warm Rerun)** baseline:

| Page | PyTorch MPS (Warm Rerun) (s) | Metal Seq (Optimized) (s) | Speedup (Seq vs MPS) | Metal Concur (Optimized) (s) | Speedup (Concur vs MPS) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Page 224 | 118.81s | 24.27s | 4.89x | 33.39s | 3.56x |
| Page 234 | 157.22s | 32.94s | 4.77x | 38.40s | 4.09x |
| Page 237 | 169.46s | 29.10s | 5.82x | 43.79s | 3.87x |
| Page 241 | 75.94s | 25.42s | 2.99x | 50.03s | 1.52x |
| Page 244 | 65.08s | 22.28s | 2.92x | 60.99s | 1.07x |
| Page 247 | 51.55s | 30.37s | 1.70x | 45.89s | 1.12x |
| Page 258 | 25.60s | 15.82s | 1.62x | 61.95s | 0.41x |
| Page 281 | 17.88s | 14.50s | 1.23x | 65.40s | 0.27x |
| Page 303 | 15.18s | 16.49s | 0.92x | 49.01s | 0.31x |
| Page 334 | 15.62s | 15.33s | 1.02x | 48.40s | 0.32x |
| **Total** | **712.34s** | **226.53s** | **3.14x** | **497.25s** | **1.43x** |
| **Mean** | **71.23s** | **22.65s** | **3.14x** | **49.72s** | **1.43x** |

- **Sequential Acceleration**: The optimized sequential Metal engine is **3.14x faster** than PyTorch MPS.
- **Parity Status**: 100% bit-exact layout extraction correctness parity is achieved.

---

## 3. Next Steps & Future Plans

1. **Vision QKV 2SG Probe**:
   - **Context**: The 2SG pattern produced a stable win for generic dense and QuickGELU dense. QKV still costs about `0.84s/page` across 32 vision blocks.
   - **Handoff Task**: Apply the same two-simdgroup weight-sharing pattern to `mu_dense_bf16_bias_rows_simdgroup_qkv` behind an escape hatch first. Promote only if page 224/244/303 split profiles show a stable win and trace parity holds.
2. **Full 10-Page Baseline Refresh**:
   - **Context**: The 10-page table above predates the Vision 2SG dense default.
   - **Handoff Task**: Re-run the 10-page optimized sequential baseline with `--kv-cache-bf16 --use-icb` after the current commit if a clean machine window is available. Avoid concurrent benchmark conclusions until worker contention is re-profiled.
3. **Push/Publish Local Commits**:
   - **Handoff Task**: Push the local branch `codex/mineru-metal-backend` after this handoff/update commit if remote publication is desired.
