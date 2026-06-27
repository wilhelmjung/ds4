# Outstanding Metal Backend Optimization Opportunities

This document outlines the remaining optimization areas and potential future tasks for the MinerU Metal backend. Following the successful implementation of the multi-page pipelined engine, Vision/Decoder QKV projection fusion, O-proj/Down-proj residual add fusion, MPSGraph SDPA, and SIMD LayerNorm, the native Metal backend is now **1.59x faster** than the warm PyTorch/MPS reference.

---

## Completed Optimizations (June 2026)

*   **Vectorized Memory Loading (GEMV)**: Casting weights to `ushort4`/`float4` for SIMD reduction, boosting loading efficiency by 2x.
*   **FlashAttention Text Prefill Stage**: Completed and enabled by default, reducing prefill latency significantly.
*   **Engine Re-use & Pipelining**: Refactored the native C CLI and benchmark script to reuse a single engine lifecycle, avoiding startup/tokenizer parsing overhead.
*   **Vision Attention MPSGraph Promotion**: Fused Q/K/V vision attention into Apple's MPSGraph SDPA execution, reducing layout vision encode time to ~7.4s.
*   **Vision LayerNorm SIMD Promotion**: Replaced scalar row-wise LayerNorm with a cooperative SIMD LayerNorm, achieving a 1.7x speedup on total vision encode.
*   **QKV & Residual Add Fusion**: Fused the independent Q, K, and V projection dispatches in both Vision Tower and Decoder, and combined O-proj/Down-proj with their residual additions (`mu_gpu_dense_probe_add_ctx`).

---

## Future Optimization Opportunities

### 1. Decoder Dispatch Count & Indirect Command Buffers (ICB)

*   **Current State**: Each auto-regressive decode token generation step requires 145 kernel dispatches across the 24 decoder layers. This introduces significant host-device enqueueing and driver scheduling overhead on the CPU side.
*   **Opportunity**: Implement Metal Indirect Command Buffers (ICB) or construct a single aggregated command buffer to encode the entire multi-layer decoding loop on the GPU side. Alternatively, implement a coarser fused decoder kernel that schedules multiple attention and MLP layers directly.
*   **Expected Benefit**: Substantially lower CPU usage, reduced command queue starvation, and a 1.15x–1.30x speedup in the `text_generate_decode` stage.

### 2. Vision FFN Module Fusion (FC1 + GELU + FC2)

*   **Current State**: In the Vision Tower, the FFN blocks consist of two separate projection GEMVs (`fc1` and `fc2`) and a GELU activation, dispatched separately. The split profile shows `fc1_gelu + fc2` takes ~3.09s, representing ~42% of the vision non-attention bottleneck.
*   **Opportunity**: Fuse FC1, GELU activation, and FC2 down-projection into a single combined kernel. Qwen2-VL's intermediate FFN activations can be kept in threadgroup memory or registers rather than writing/reading from global VRAM.
*   **Expected Benefit**: Eliminates intermediate VRAM roundtrips for Vision MLP features, reducing `layout_vision_encode` and `content_region_vision_encode` times further.

### 3. High-Value MPSGraph GEMM Integrations

*   **Current State**: Currently, custom MSL SIMD-group matrix multiply kernels are used for major projections. 
*   **Opportunity**: Integrate Apple's optimized `MPSGraph` matrix multiplication or MPS Matrix Multiplication library (`MPSMatrixMultiplication`) for large static projection shapes in the attention and MLP layers where the shape remains constant across steps.
*   **Expected Benefit**: Exploits hardware-specific Apple Silicon AMX (Apple Matrix Coprocessor) acceleration more efficiently than hand-written MSL shader loops, especially on table-heavy documents.
