# Outstanding Metal Backend Optimization Opportunities

This document outlines the remaining optimization areas and potential future tasks for the MinerU Metal backend after the successful implementation of Phase 6.

---

## 1. Vectorized Memory Loading in Text Decode Projections (GEMV)

*   **Current State**: Autoregressive text generation/decoding (batch size/sequence length = 1) dispatches `mu_dense_f32_bias_probe_simd` inside `mu_text_cached_step` for Query, Key, Value, and Output projection layers. Inside the shader, weights `w` are loaded individually as `ushort` scalar values.
*   **Opportunity**: The hidden dimension is 896, which is a multiple of 8. We can optimize this by casting weights to `ushort4 *` or `ushort8 *` to load 16-byte/32-byte chunks.
*   **Expected Benefit**: Doubled memory loading efficiency and improved memory coalescing for the 4 dense layers per text decoder layer, lowering memory bandwidth bottlenecks during autoregressive decoding.

---

## 2. FlashAttention in Text Decoder Prefill Stage

*   **Current State**: Fully fused FlashAttention (`mu_vision_attn_rows_flash`) is only active in the 32-layer vision tower. The 24-layer text decoder prefill stage still relies on separate attention projection and softmax kernels.
*   **Opportunity**: Extend the FlashAttention kernel design to support the text decoder prefill sequence processing, fusing QK projection, softmax, and PV multiplication.
*   **Expected Benefit**: Massive reduction in global VRAM allocation and memory bandwidth pressure during the prompt prefill stage for long input sequences.

---

## 3. SwiGLU / FFN Fusion for Text Decoder Prefill Stage

*   **Current State**: Phase 6 implemented FFN fusion (`mu_text_decode_fused_ffn`) specifically for the autoregressive decode step (sequence length = 1) where computations fit inside threadgroup shared memory. The prefill stage (sequence length > 1) still calls Gate, Up, and Down projections as separate dense layers.
*   **Opportunity**: Implement a tiled matrix-multiplication FFN (SwiGLU) fusion kernel designed for sequence lengths $N > 1$.
*   **Expected Benefit**: Eliminates the intermediate Gate/Up projection activation VRAM writes and reads for the prefill stage.

---

## 4. CPU-GPU Pipeline Overlapping (Page-Level Pipelining)

*   **Current State**: MinerU processes pages sequentially. The host CPU thread commits a command buffer for page $N$ and blocks via `mu_gpu_cmd_commit_and_wait` or `waitUntilCompleted` before starting the pre-processing (such as image crop decoding and patch embedding) for page $N+1$.
*   **Opportunity**: Implement double or triple command buffering at the page level. While the GPU is busy executing the vision tower and text generation for page $N$, the CPU should begin decoding and patch-embedding page $N+1$.
*   **Expected Benefit**: Overlaps CPU-heavy preprocessing with GPU-heavy inference, increasing overall page-per-second throughput in multi-page documents.

---

## 5. Register Pressure & Occupancy Tuning for FlashAttention

*   **Current State**: The `mu_vision_attn_rows_flash` kernel processes tiles of size $32 \times 80$. This large block size allocates a significant amount of threadgroup memory and registers, which can limit warp occupancy per SIMD-core on older Apple Silicon GPUs.
*   **Opportunity**: Profile and tune the tile dimensions (e.g., trying $16 \times 80$ or $16 \times 64$) or utilize SIMD-group shuffle instructions for the softmax stats collection to reduce shared memory usage.
*   **Expected Benefit**: Higher GPU core occupancy and faster attention execution times on lower-tier Apple Silicon chips.
