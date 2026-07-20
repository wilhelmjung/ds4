---
name: mineru-optimization-guide
description: Guidelines, architectures, and optimization patterns for MinerU Metal/C inference engine, detailing kernel fusions and concurrency rules.
---

# MinerU Metal Optimization Guide

This guide compiles key architectural patterns and lessons learned during the optimization of the MinerU Metal/C engine.

## 1. Custom MSL Shaders vs. MPSMatrixMultiplication
- **Rule**: Prefer custom MSL `simdgroup_matrix` GEMM kernels over `MPSMatrixMultiplication` for intermediate matrix multiplications in the vision tower or text decoder blocks.
- **Why**: MPS Matrix Multiplication requires separate command encoder switches (ending the current encoder, encoding MPS, starting a new encoder). For shapes like width-1280 vision block multiplications, the driver CPU overhead and pipeline bubbles of 96+ encoder switches per page completely negate the AMX hardware acceleration. Keeping operations inside a single contiguous compute command encoder is faster.

## 2. Kernel Fusion Design
- **GQA Attention**: Fused FlashAttention (`mu_vision_attn_rows_flash` and `mu_text_prefill_attn_flash`) must compute online softmax statistics dynamically to avoid costly intermediate QK transpose and softmax allocations.
- **FFN SwiGLU**: Fuse gate projection, up projection, SiLU activation, and element-wise multiplication into a single SIMD-group unified kernel (`mu_dense_bf16_rows_simdgroup_swiglu`). This eliminates 2 dispatches per layer and removes intermediate buffers.
- **Residual Addition**: Fuse output projections directly with the residual summation to save intermediate VRAM writes/reads.
- **Vision Dense/QKV 2SG**: Prefer the two-simdgroup row kernels for the established Vision dense shapes, including fused QKV projection. `QKV` is the fused Query/Key/Value projection; `2SG` means one threadgroup runs two simdgroups, computes two 8-row tiles, and shares the same loaded weight tile. Keep `MU_DENSE_ROWS_NO_2SG=1` as the single escape hatch for 1SG A/B checks.

## 3. Concurrency and Thread Safety
- **Rule**: When executing multiple worker threads via `--threads > 1`, keep all shared buffers and caches thread-safe.
- **Weight Cache**: Protect the global `weight_cache` lookups and allocations in `mu_metal.m` using a `pthread_mutex_t`.
- **Weight-Cache Allocation Throttle**: Do not move `newBufferWithBytes*` weight-cache misses outside the global mutex by default. A lock-free miss path with a pending/condition-variable guard removed duplicate allocations, but regressed the pages `224,244,303` probe to `75.911148s` profiled and `108.154561s` no-profile wall time. Treat the mutex as both synchronization and cold-cache copy throttle unless a fresh adjacent benchmark proves otherwise.
- **Concurrency Profiling**: Use `MU_CONCURRENCY_PROFILE=1` for `--threads > 1` work. It records page worker events, Metal `command_wait` timings, and weight-cache wait/hold/copy stats in page-scoped logs. Always pair it with a no-profile wall-clock artifact before promoting a concurrency change.
- **Cold/Warm Control**: Use `mineru/tests/mu_benchmark_pages.py --warmup-runs 1` or reverse-order A/B for concurrency probes. A cold/noisy default run previously measured `120.556958s`, while the same default after a warm run measured `60.013088s`; do not promote a concurrency result from one-way adjacent timing alone.
- **Rejected Command-Queue Probe**: Do not add a per-worker `MTLCommandQueue` path from the current evidence. An opt-in `MU_METAL_MULTI_QUEUE=1` probe on pages `224,244,303` was slower than adjacent default (`134.306501s` vs `125.483086s` wall), so single-queue serialization is not solved by simply creating more command queues.
- **Rejected Prefetch Toggle**: Do not disable the background prefetch thread by default. A first-order run looked faster only because the default baseline was cold/noisy; reverse-order checking showed warm default slightly faster than disabling prefetch (`60.013088s` vs `62.323259s` wall).
- **Rejected Prefetch Patch-Embed Probe**: Do not force CPU patch embedding in the background prefetch thread from the current evidence. A temporary opt-in probe on pages `224,244,303` measured `42.214261s`, while the reverse-order default measured `41.852267s`; the probe was deleted.
- **MPS Objects**: Metal Performance Shaders (MPS) kernel objects (such as `MPSMatrixMultiplication`) are **not thread-safe** for concurrent `encodeToCommandBuffer` calls. When `MU_CONCURRENT_WORKERS > 1`, bypass the kernel cache and allocate a fresh MPS kernel object per-call.

## 4. Occupancy and Register Pressure Tuning
- **Rule**: Minimize per-thread local arrays (registers) in complex kernels like FlashAttention, shifting persistent thread variables to threadgroup shared memory if necessary.
- **Why**: Thread register counts exceeding 128 on Apple Silicon drop GPU compute core occupancy significantly. Buffering Query (Q) vectors in shared memory and tuning Key-Value tile sizes (e.g. step=16 instead of step=32) reduces threadgroup storage footprint and halves per-thread registers, improving overall warp occupancy and delivering a **1.05x-1.07x speedup** on memory/bandwidth-bound kernels.

## 5. KV-Cache BF16 Compression
- **Rule**: Store key (`k_cache`) and value (`v_cache`) caches in BF16 (`ushort`) precision instead of FP32 (`float`) during the text prefill and decoding phases under env toggle `MU_KV_CACHE_BF16=1`.
- **Why**: Halves the VRAM memory footprint and memory bandwidth usage for KV cache lookups. For bandwidth-bound text decoding, this reduces E2E inference times (achieved an **~8.8% E2E speedup** on layout page extraction).
- **Implementation Detail**: Round FP32 values to BF16 (nearest even) during storage using a fast bitwise operation `(bits >> 16)` with rounding offset. Read and convert back dynamically inside the shader using the dynamic loader helper `mu_load_cache`. To avoid regression trace discrepancies, keep it optional via the toggle.

## 6. Indirect Command Buffers (ICB) for Autoregressive Decoding
- **Rule**: For repetitive, multi-kernel dispatch sequences (like VLM/LLM autoregressive decoding steps), pre-record all compute kernel dispatches, pipeline bindings, and static resources into a single `MTLIndirectCommandBuffer` during the first step, and execute it using `executeCommandsInBuffer:withRange:` under toggle `MU_TEXT_DECODE_ICB=1`.
- **Dynamic Parameters**: Since `MTLIndirectComputeCommand` does not support binding dynamic host variables directly via `setBytes:`, pack all dynamic values (e.g. `pos3`, `cache_pos`, `cache_len`, `use_bf16_cache`) into a unified struct `mu_gpu_decode_dynamic_params` mapped to a shared buffer, updating the buffer values on CPU before executing the ICB.
- **Compute Barriers**: By default, compute commands in an ICB are dispatched concurrently. To enforce sequential execution order and prevent data hazards between dependent layers, call `[cmd setBarrier]` sequentially on all indirect commands (except the first one).
- **Pipeline Setup**: Ensure all target compute pipelines are created using `MTLComputePipelineDescriptor` with `supportIndirectCommandBuffers = YES` enabled, otherwise calling `setComputePipelineState:` on an indirect command will crash (EXC_BAD_ACCESS).

## 7. Current 10-Page Metal Baseline
- **Rule**: Use `/tmp/mu_10page_seq_qkv2sg_refresh.json` as the single-page latency baseline and `/tmp/mu_10page_threads2_qkv2sg_refresh.json` as the batch-throughput baseline after Vision dense 2SG, QKV 2SG, BF16 KV cache, and decode ICB defaults.
- **Numbers**: Sequential completed `10/10` pages with `0` failures and `0` fallback rows in `174.984202s` summed page time (`17.498420s/page`). `--threads 2` completed the same gate in `126.861503s` wall time with `24.487243s` mean page latency, a `1.38x` throughput speedup over sequential and `5.62x` over the existing PyTorch/MPS warm-rerun reference.
- **Concurrency Scope**: `--threads 2` is the current throughput recommendation; `--threads 4` was slower than `--threads 2` on the pages `224,244,303` probe. Before raising concurrency, profile shared GPU scratch, command queue serialization, weight-cache locking, and ICB overlap.
- **Rejected Concurrency Direction**: Weight-cache lock-free allocation, per-worker command queues, disabling background prefetch, and forcing CPU patch embedding in background prefetch are already measured and rejected. The next concurrency pass should focus on ICB decode overlap and shared scratch pressure, with cold/warm benchmark control enforced before promotion.

## 8. Single-Pass Online Softmax & RoPE Inv-Freq Precomputation (pmetal-inspired Attention)
- **Rule**: Replace 3-pass softmax loops (max -> sum(exp) -> acc) in cached text attention kernels (`mu_text_attn_cached`, `mu_text_attn_cached_simd`, `mu_text_attn_cached_simd_batched`) with single-pass Online Softmax (Welford/Rescaling), and use precomputed `mu_text_rope_inv_freqs[32]` constant tables for RoPE inverse frequency calculations.
- **Why**:
  - **Memory Bandwidth**: Eliminates 2 out of 3 full traversals over the KV-Cache per decoding step, reducing KV-Cache memory reads by 66%.
  - **Occupancy & Zero Barriers**: Eliminates `threadgroup float scores[4096]` allocation and `threadgroup_barrier` in SIMD-group cached attention kernels, freeing threadgroup memory and removing pipeline stalls.
  - **ALU Efficiency**: Eliminates redundant `pow(1000000.0f, ...)` calculations in RoPE loops by converting to inline constant table lookup.
- **Performance Impact**: Improved 10-page warm-run wall-clock time to **222.48s** (**22.25s/page**), achieving **10.70x** speedup over CPU baseline and **1.52x** over PyTorch MPS warm baseline.

