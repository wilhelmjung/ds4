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

## 3. Concurrency and Thread Safety
- **Rule**: When executing multiple worker threads via `--threads > 1`, keep all shared buffers and caches thread-safe.
- **Weight Cache**: Protect the global `weight_cache` lookups and allocations in `mu_metal.m` using a `pthread_mutex_t`.
- **MPS Objects**: Metal Performance Shaders (MPS) kernel objects (such as `MPSMatrixMultiplication`) are **not thread-safe** for concurrent `encodeToCommandBuffer` calls. When `MU_CONCURRENT_WORKERS > 1`, bypass the kernel cache and allocate a fresh MPS kernel object per-call.

## 4. Occupancy and Register Pressure Tuning
- **Rule**: Minimize per-thread local arrays (registers) in complex kernels like FlashAttention, shifting persistent thread variables to threadgroup shared memory if necessary.
- **Why**: Thread register counts exceeding 128 on Apple Silicon drop GPU compute core occupancy significantly. Buffering Query (Q) vectors in shared memory and tuning Key-Value tile sizes (e.g. step=16 instead of step=32) reduces threadgroup storage footprint and halves per-thread registers, improving overall warp occupancy and delivering a **1.05x-1.07x speedup** on memory/bandwidth-bound kernels.

## 5. KV-Cache BF16 Compression
- **Rule**: Store key (`k_cache`) and value (`v_cache`) caches in BF16 (`ushort`) precision instead of FP32 (`float`) during the text prefill and decoding phases under env toggle `MU_KV_CACHE_BF16=1`.
- **Why**: Halves the VRAM memory footprint and memory bandwidth usage for KV cache lookups. For bandwidth-bound text decoding, this reduces E2E inference times (achieved an **~8.8% E2E speedup** on layout page extraction).
- **Implementation Detail**: Round FP32 values to BF16 (nearest even) during storage using a fast bitwise operation `(bits >> 16)` with rounding offset. Read and convert back dynamically inside the shader using the dynamic loader helper `mu_load_cache`. To avoid regression trace discrepancies, keep it optional via the toggle.

