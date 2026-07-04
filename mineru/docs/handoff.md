# Handoff Report: Metal Backend Optimizations & Performance Baseline

## 1. Accomplished Work (Phases 23 & 24)

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
- Fixed the ICB-recorded `MU_TEXT_DECODE_QKV_ROPE_FUSION=1` command: dynamic parameters now bind in shader order (`pos3`, `cache_pos`, `cols`, `use_bf16_cache`), and the fused dispatch now covers the actual 640 output rows instead of 1152.
- Result: the previous `layout` trace generation mismatch under `--kv-cache-bf16 --use-icb` plus `MU_TEXT_DECODE_QKV_ROPE_FUSION=1` is fixed. The path is correct but still not promoted because the fresh layout trace decode time was slower than default ICB (`0.2101s` vs `0.1858s`).
- Build warning cleanup: removed an unused `name` variable in the batched decode path.

### Decoder FFN SIMDGroup Default
- Replaced the non-ICB default decoder FFN path with the existing prefill SIMDGroup SwiGLU helper plus `dense_f32_rows` down projection and residual add.
- Kept the old monolithic `mu_text_decode_fused_ffn` path behind `MU_TEXT_DECODE_FFN_NO_SIMDGROUP=1`; `MU_TEXT_NO_FUSED_FFN=1` still forces the fully unfused fallback.
- Fresh `layout` trace A/B with `--kv-cache-bf16`:
  - New default: decode `0.1091s`, trace parity OK.
  - Old path via `MU_TEXT_DECODE_FFN_NO_SIMDGROUP=1`: decode `0.1459s`, trace parity OK.
  - ICB path remains correct (`--use-icb`: decode `0.1601s`) but still records the old monolithic FFN command and does not use the new host-side FFN sequence.

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

1. **Decoder ICB / Attention Follow-up**:
   - **Context**: The non-ICB decoder FFN now has a faster default path. ICB remains correct but misses that FFN improvement because it still records the old monolithic FFN kernel. QKV/RoPE fusion is correctness-fixed but not fast enough to promote.
   - **Handoff Task**: Either record an ICB-compatible FFN sequence using RMSNorm + SIMDGroup SwiGLU + shader down-proj + residual add, or continue with one small cached-attention tiling experiment. Gate with `layout`/`text` trace parity and adjacent timing.
2. **Re-implement Vision FFN Fusion using `simdgroup_matrix` GEMM primitives**:
   - **Context**: The existing `mu_vision_fused_ffn` kernel was disabled by default because it relied on sequential dot product loops, dropping warp occupancy and running slower than the unfused SIMD path.
   - **Handoff Task**: Re-architect `mu_vision_fused_ffn.metal` using MSL's cooperative matrix multiplication (`simdgroup_matrix`) only after decoder experiments stop showing cheap wins.
3. **Push/Publish Local Commits**:
   - **Handoff Task**: Push the local branch `codex/mineru-metal-backend` (currently 20 commits ahead) to remote repository.
