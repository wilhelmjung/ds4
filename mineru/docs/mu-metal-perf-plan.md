# MinerU Metal 性能优化执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `mu` 引擎的 Metal 后端从当前的正确性桥接（Correctness Bridge）优化为生产级性能引擎，最终目标是匹配或超越 `transformers/mps` 的推理速度，同时保持与 CPU 参考路径完全一致的精度。

**Source Documents:**
- [mu-performance-optimization.md](file:///Users/will/github/ds4/mineru/docs/mu-performance-optimization.md) — 性能优化设计方案
- [mu-architecture.md](file:///Users/will/github/ds4/mineru/docs/mu-architecture.md) — 模型架构文档
- [mu-metal-design.md](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md) — Metal 后端设计（核心约束来源）
- [mu-metal-plan.md](file:///Users/will/github/ds4/mineru/docs/mu-metal-plan.md) — Metal 后端实现计划（已完成的实现里程碑记录）
- [mu-performance-report.md](file:///Users/will/github/ds4/mineru/docs/mu-performance-report.md) — 历史性能数据与瓶颈分析
- [mu-design.md](file:///Users/will/github/ds4/mineru/docs/mu-design.md) — 引擎整体设计
- [mu-skills.md](file:///Users/will/github/ds4/mineru/docs/mu-skills.md) — 可复用工程实践

**Architecture Constraints (from [mu-design.md](file:///Users/will/github/ds4/mineru/docs/mu-design.md) and [mu-metal-design.md](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md)):**
- 不要泛化或改造 `ds4.c` 为多模型运行时，MinerU 引擎独立在 `mineru/` 下
- CPU 始终是默认后端和精度权威，Metal 不得改变 CPU 代码路径的数值行为
- Objective-C 限定在 `mu_metal.m`，`mu.c` 不引入 Metal 头文件
- 后端选择在 **stage 级别** 做 dispatch（vision encode / text generate），不要在每个 math helper 里散落后端判断
- 不要在第一个 Metal 版本中做权重量化

**Tech Stack:** C99, Objective-C ARC, Apple Metal, BF16 safetensors mmap weights, Accelerate/CBLAS CPU reference, Python smoke harness.

---

## Current State

来自 [mu-metal-plan.md](file:///Users/will/github/ds4/mineru/docs/mu-metal-plan.md) 和 [mu-performance-report.md](file:///Users/will/github/ds4/mineru/docs/mu-performance-report.md) 的已完成里程碑：

| Milestone | Status |
| --- | --- |
| Backend Shell (`mu_gpu.h`, `mu_metal.m`, `metal/`) | ✅ Done |
| Dense & Norm Kernels (BF16 probe/rows) | ✅ Done |
| Vision Tower Metal (32 ViT blocks + merger) | ✅ Done |
| Text Decoder Prefill Metal | ✅ Done |
| Text Decoder KV-Cache Decode Metal | ✅ Done |
| End-to-End 10-page Benchmark | ✅ Done |

## Current Performance Baseline

来自 [mu-performance-report.md](file:///Users/will/github/ds4/mineru/docs/mu-performance-report.md) 的 10-page full-content512 KV-cache 基线：

| Comparison | Current |
| --- | ---: |
| Metal / CPU (10-page mean) | 2.92x slower |
| Metal / Transformers/MPS (10-page mean) | 7.05x slower |
| CPU-vs-Metal Token F1 | 1.0000 |
| CPU-vs-Metal BBox IoU | 1.0000 |
| CPU-vs-Metal Table cells | 104/104 |

### Stage-Level Bottleneck Distribution (10-page mean, from [performance report](file:///Users/will/github/ds4/mineru/docs/mu-performance-report.md#L827-L836))

| Stage | CPU mean s/page | Metal mean s/page | Metal / CPU |
| --- | ---: | ---: | ---: |
| layout_vision_encode | 42.24 | 105.94 | **2.51x** |
| layout_generate | 8.51 | 27.57 | 3.24x |
| content_region_vision_encode | 19.81 | 51.74 | 2.61x |
| content_region_generate | 18.17 | 82.69 | **4.55x** |
| content_total | 38.17 | 134.61 | 3.53x |
| page_total | 92.95 | 271.94 | 2.93x |

> [!IMPORTANT]
> **Vision encode** 占 Metal 总时间的 58%（`105.94 + 51.74 = 157.68s` / `271.94s`）。
> **Text generate** 占 Metal 总时间的 41%（`27.57 + 82.69 = 110.26s` / `271.94s`）。
> 两者的根因一致：**每次 GPU 算子调用时重复创建/拷贝 MTLBuffer + 逐算子 waitUntilCompleted 阻塞**。

### Anti-pattern: 当前 mu_metal.m 的调度模式

来自 [mu-performance-optimization.md §1](file:///Users/will/github/ds4/mineru/docs/mu-performance-optimization.md#L7-L19) 的瓶颈诊断：

```text
当前每个 GPU 算子内部独立执行：
  1. newBufferWithBytes(x)      ← CPU→GPU memcpy 输入
  2. newBufferWithBytes(w_bf16)  ← CPU→GPU memcpy 权重（每次！）
  3. newBufferWithBytes(bias)    ← CPU→GPU memcpy bias（每次！）
  4. newBufferWithLength(out)    ← 分配输出
  5. newBufferWithBytes(&cols)   ← 为 4 字节标量分配 MTLBuffer
  6. createCommandBuffer → createEncoder → encode → commit → waitUntilCompleted
  7. memcpy(out, [out_buf contents])  ← GPU→CPU 读回
```

一个 Vision Block 执行约 10 个这样的算子 × 32 层 = **320 次 buffer 创建 + 320 次 wait**。
一个 Text Decoder 层执行约 13 个算子 × 24 层 = **312 次 buffer 创建 + 312 次 wait**。

---

## Required Verification Order

来自 [mu-metal-design.md §Parity Gates](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L311-L348) 和 [mu-metal-plan.md §Verification Order](file:///Users/will/github/ds4/mineru/docs/mu-metal-plan.md#L52-L77)：

**每个 Task 完成后，必须按以下顺序执行验证。CPU 先于 Metal：**

```bash
# 1. CPU 回归门禁（必须先跑）
make mu-test
make -B mu
./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json

# 2. Metal 精度门禁
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json

# 3. 单页性能门禁（Phase 完成后选择性运行）
MU_TIMING=1 /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal --pages 224 --max-new-tokens 128 \
  --content-max-new-tokens 512 --timeout 7200 --timing --resume \
  --out /tmp/mu-benchmark-metal-page224-opt.json \
  --save-output-dir /tmp/mu-opt-page224

# 4. 精度对比门禁
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_compare_outputs.py \
  --ref-json /tmp/mu-fullcontent512-kvcache-page224/cpu_page_0224.json \
  --pred-json /tmp/mu-opt-page224/metal_page_0224.json \
  --out /tmp/mu-opt-page224/cpu-vs-metal.metrics.json
```

**精度红线**（来自 [mu-skills.md](file:///Users/will/github/ds4/mineru/docs/mu-skills.md#L30-L43)）：

| Metric | Required |
| --- | ---: |
| CPU-vs-Metal exact block-count pages | 10/10 |
| CPU-vs-Metal ordered type accuracy | 1.0000 |
| CPU-vs-Metal ordered bbox IoU | 1.0000 |
| CPU-vs-Metal content token F1 | 1.0000 |
| CPU-vs-Metal table exact cell recall | 1.0000 (104/104) |

---

## Phase 1: 持久化权重缓冲区

> 消除每次算子调用时 `newBufferWithBytes` 重复拷贝 **静态模型权重** 的开销。
> 设计来源：[mu-performance-optimization.md §2A](file:///Users/will/github/ds4/mineru/docs/mu-performance-optimization.md#L26-L32)，[mu-metal-design.md §Weight And Buffer Strategy](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L172-L194)。

### Task 1.1: 在 `mu_gpu` 中添加权重缓冲区 Lazy Cache

**Files:**
- Modify: [mu_gpu.h](file:///Users/will/github/ds4/mineru/mu_gpu.h)
- Modify: [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m)

**Design Note:** [mu-metal-design.md](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L177-L194) 规定采用 **lazy staging** 策略——首次需要时创建 MTLBuffer 并缓存，而非 engine open 时一次性上传全部权重。这保持了 CPU 访问能力，避免启动时内存峰值。

以 safetensors mmap 中每个 tensor 的 CPU 起始指针地址为 key（因为 mmap 中每个 tensor 的地址是唯一且稳定的）。

- [ ] **Step 1: 在 `mu_gpu` struct 中添加缓冲区关联数组**

```c
// mu_metal.m 内部
#define MU_GPU_WEIGHT_CACHE_CAP 1024

typedef struct {
    const void *cpu_ptr;    // mmap 中 tensor 数据起始地址
    NSUInteger length;
    id<MTLBuffer> buffer;
} mu_gpu_cached_buffer;

// 在 struct mu_gpu 中添加：
mu_gpu_cached_buffer weight_cache[MU_GPU_WEIGHT_CACHE_CAP];
int weight_cache_count;
```

- [ ] **Step 2: 实现 `mu_gpu_get_or_create_buffer` 内部函数**

查找已缓存的 buffer 或首次创建。对于 16KB 页对齐的指针，优先使用 `newBufferWithBytesNoCopy` 实现零拷贝（[mu-performance-optimization.md §4C](file:///Users/will/github/ds4/mineru/docs/mu-performance-optimization.md#L128-L132)）。

- [ ] **Step 3: 在 `mu_gpu_destroy` 中释放所有缓存**
- [ ] **Step 4: 编译验证 `make -B mu && make mu-test`**

### Task 1.2: 在所有 GPU 算子中使用缓存权重

**Files:**
- Modify: [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m)

**范围：** 只替换 **静态权重和 bias** 的 buffer 创建。输入 `x` 和输出 `out` 仍用临时 buffer（Phase 2 处理）。

需要修改的 12 个函数（来自 [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m) 的 `newBufferWithBytes:w_bf16` 和 `newBufferWithBytes:bias_bf16` 调用）：

1. `mu_gpu_dense_probe` — w_buf
2. `mu_gpu_dense_bf16_bias_probe` — w_buf, bias_buf
3. `mu_gpu_dense_f32_bias_probe` — w_buf, bias_buf
4. `mu_gpu_dense_f32_rows` — w_buf
5. `mu_gpu_dense_f32_bias_rows` — w_buf, bias_buf
6. `mu_gpu_dense_bf16_bias_rows` — w_buf, bias_buf
7. `mu_gpu_rmsnorm_bf16_probe` — w_buf
8. `mu_gpu_rmsnorm_bf16_rows` — w_buf
9. `mu_gpu_layernorm_bf16_probe` — w_buf, bias_buf
10. `mu_gpu_layernorm_bf16_rows` — w_buf, bias_buf

- [ ] **Step 1: 逐函数替换 `newBufferWithBytes` 为 `mu_gpu_get_or_create_buffer`**
- [ ] **Step 2: 运行全部验证门禁**
- [ ] **Step 3: 提交**

### Task 1.3: 添加诊断计数器

**Files:**
- Modify: [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m), [mu_gpu.h](file:///Users/will/github/ds4/mineru/mu_gpu.h)

遵循 [mu-metal-design.md](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L164-L168) 要求的 `optional debug counters for CPU fallback and bytes moved`。

- [ ] **Step 1: 添加 `buffer_cache_hits` / `buffer_cache_misses` / `buffer_new_allocs` 计数器**
- [ ] **Step 2: 在 `MU_METAL_DEBUG` 模式下打印统计**
- [ ] **Step 3: 运行 page 224 验证缓存命中率**

期望：`misses` 应等于唯一权重 tensor 数量（约 681 个），后续全部命中。

- [ ] **Step 4: 提交**

### Task 1.4: Phase 1 性能基线

- [ ] **Step 1: 运行 page 224 单页 benchmark**
- [ ] **Step 2: 记录对比数据到 [mu-performance-report.md](file:///Users/will/github/ds4/mineru/docs/mu-performance-report.md)**

---

## Phase 2: 消除标量 & 临时 Buffer 分配开销

> 消除每次算子调用时为标量参数和临时 activation 创建 MTLBuffer 的开销。

### Task 2.1: 将标量参数改用 `setBytes`

**Files:**
- Modify: [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m)

**动机：** 当前每个算子为 4 字节的 `int cols`、`int n`、`float eps` 等标量分配独立的 `MTLBuffer`。Metal API 的 `setBytes:length:atIndex:` 可以直接将小常量嵌入 Command Encoder，无需 buffer 分配。

- [ ] **Step 1: 将全部 `cols_buf`、`n_buf`、`eps_buf`、`out_cols_buf`、`groups_buf`、`cache_len_buf`、`seq_buf` 替换为 `setBytes`**

将：
```objc
id<MTLBuffer> cols_buf = [gpu->device newBufferWithBytes:&cols
                                                  length:sizeof(cols)
                                                 options:MTLResourceStorageModeShared];
[encoder setBuffer:cols_buf offset:0 atIndex:3];
```
替换为：
```objc
[encoder setBytes:&cols length:sizeof(cols) atIndex:3];
```

受影响函数：全部约 20 个 `mu_gpu_*` 函数。

- [ ] **Step 2: 门禁验证**
- [ ] **Step 3: 提交**

### Task 2.2: 实现 Scratchpad Activation Arena

**Files:**
- Modify: [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m)

**动机：** 来自 [mu-metal-design.md §Later optimization](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L186-L190)："Add an activation arena or `MTLHeap` after kernel shapes stabilize."。现在 kernel 形状已稳定，可以实施。

- [ ] **Step 1: 在 `mu_gpu` 中预分配 Scratch 缓冲区**

```c
id<MTLBuffer> scratch_a;  // Ping buffer
id<MTLBuffer> scratch_b;  // Pong buffer
NSUInteger scratch_size;   // 取 max(vision, text) 的上界
```

Vision 最大中间值：`rows × 5120 × sizeof(float)`（spatial merger 输出）。
Text 最大中间值：`inter × sizeof(float)` = `4864 × 4` = 19,456 bytes。
预分配时按 vision 上界设置。

- [ ] **Step 2: 改造一个算子验证方案 — 从 `mu_gpu_dense_probe` 开始**
- [ ] **Step 3: 推广到全部 probe / rows 算子**
- [ ] **Step 4: 门禁验证**
- [ ] **Step 5: 提交**

### Task 2.3: Phase 2 性能基线

- [ ] **Step 1: Page 224 benchmark**
- [ ] **Step 2: 记录对比数据**

---

## Phase 3: 异步 Command Buffer 管线

> 消除每个 GPU 算子调用后 `waitUntilCompleted` 带来的 CPU 阻塞开销。
> 设计来源：[mu-performance-optimization.md §2B](file:///Users/will/github/ds4/mineru/docs/mu-performance-optimization.md#L34-L43)。

```text
Current:   [CPU] → Submit Kernel → [GPU] Run → Wait → [CPU] → Submit Kernel → ...
Optimized: [CPU] → Encode All Kernels → Commit → [GPU] Parallel execution → Wait (only at end)
```

### Task 3.1: 引入 `mu_gpu_cmd_ctx` 批量编码接口

**Files:**
- Modify: [mu_gpu.h](file:///Users/will/github/ds4/mineru/mu_gpu.h)
- Modify: [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m)

**Design Note:** 遵循 [mu-metal-design.md §Dispatch Boundary](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L269-L289) 的原则——在 stage 边界 dispatch，而不是在每个 math helper 内部。命令上下文允许在一个 stage（如一层 decoder）内编码全部 kernel，最后一次 commit+wait。

- [ ] **Step 1: 在 `mu_gpu.h` 中声明命令上下文接口**

```c
typedef struct mu_gpu_cmd_ctx mu_gpu_cmd_ctx;
int mu_gpu_cmd_begin(mu_gpu *gpu, mu_gpu_cmd_ctx **ctx);
int mu_gpu_cmd_commit_and_wait(mu_gpu_cmd_ctx *ctx);
void mu_gpu_cmd_discard(mu_gpu_cmd_ctx *ctx);
```

- [ ] **Step 2: 在 `mu_metal.m` 中实现命令上下文**

`mu_gpu_cmd_begin`: 创建 command buffer 和 encoder。
`mu_gpu_cmd_commit_and_wait`: endEncoding → commit → waitUntilCompleted。
`mu_gpu_cmd_discard`: endEncoding 但不 commit（错误路径）。

- [ ] **Step 3: 为每个 GPU 算子添加 `_ctx` 变体**

接受 `mu_gpu_cmd_ctx *` 和 `id<MTLBuffer>` 偏移量，不创建 buffer、不 commit、不 wait，只向现有 encoder 追加 dispatch。

- [ ] **Step 4: 保留原有独立调用接口**

原有 `mu_gpu_dense_probe` 等函数改为内部：begin → _ctx 变体 → commit_and_wait。确保兼容。

- [ ] **Step 5: 门禁验证**
- [ ] **Step 6: 提交**

### Task 3.2: 在 Text Decoder 层中使用批量编码

**Files:**
- Modify: [mu.c](file:///Users/will/github/ds4/mineru/mu.c)

**动机：** Text Decoder 每层约 13 个 GPU 算子（RMSNorm → Q → K → V → Attn → O_proj → Add → RMSNorm → Gate → Up → SiLU·Mul → Down → Add）。合并到一个 cmd_ctx。

- [ ] **Step 1: 改造 `mu_text_cached_step` 的 Metal 路径使用 cmd_ctx**
- [ ] **Step 2: 验证精度不变**
- [ ] **Step 3: 提交**

### Task 3.3: 在 Vision Tower 中使用批量编码

**Files:**
- Modify: [mu.c](file:///Users/will/github/ds4/mineru/mu.c)

**动机：** Vision Tower 32 层 × 每层约 10 个 GPU 算子 = 320 次独立 dispatch。改为每层或每 N 层一个 cmd_ctx。

- [ ] **Step 1: 改造 `mu_vision_encode_hidden` 的 Metal 路径**
- [ ] **Step 2: 验证精度**
- [ ] **Step 3: 提交**

### Task 3.4: Phase 3 性能基线

- [ ] **Step 1: Page 224 benchmark**
- [ ] **Step 2: 更新 performance report**

---

## Phase 4: GPU 常驻 KV-Cache

> 将 KV-Cache 从 CPU 堆内存迁移到 GPU 缓冲区，消除 decode 阶段每步的 CPU↔GPU 拷贝。
> 设计来源：[mu-performance-optimization.md §2C](file:///Users/will/github/ds4/mineru/docs/mu-performance-optimization.md#L46-L49)。

### Task 4.1: 在 GPU 内存中分配 KV-Cache

**Files:**
- Modify: [mu.c](file:///Users/will/github/ds4/mineru/mu.c)
- Modify: [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m)
- Modify: [mu_gpu.h](file:///Users/will/github/ds4/mineru/mu_gpu.h)

**动机：** 当前 KV-Cache 在 CPU `calloc` 中（[mu.c:4842-4843](file:///Users/will/github/ds4/mineru/mu.c#L4842-L4843)），每次 `mu_gpu_text_attn_cached` 通过 `newBufferWithBytes` 拷贝整个 cache 到 GPU。24 层 × seq_len × 128 的 cache，每 decode step 传输数 MB。

- [ ] **Step 1: 添加 `mu_gpu_kv_cache` 结构体和 create/destroy**
- [ ] **Step 2: 实现 `mu_gpu_kv_cache_write_slot`（K/V 向量直写 GPU 缓冲区指定 offset）**
- [ ] **Step 3: 改造 `mu_gpu_text_attn_cached` 接受 GPU 缓冲区**
- [ ] **Step 4: 改造 `mu_text_cached_step` 和 `mu_text_prefill_cache_from_embeddings` 使用 GPU KV-Cache**
- [ ] **Step 5: 门禁验证**
- [ ] **Step 6: 提交**

---

## Phase 5: GPU 端 Argmax

> 消除每步 decode 时 logits 向量（151,936 × 4 bytes ≈ 600KB）从 GPU 读回 CPU 的开销。
> 设计来源：[mu-performance-optimization.md §2E](file:///Users/will/github/ds4/mineru/docs/mu-performance-optimization.md#L57-L61)。

### Task 5.1: 实现 Metal Argmax Kernel

**Files:**
- New: `mineru/metal/mu_sample.metal`（或扩展已有 metal 文件）
- Modify: [mu_metal.m](file:///Users/will/github/ds4/mineru/mu_metal.m), [mu_gpu.h](file:///Users/will/github/ds4/mineru/mu_gpu.h)

- [ ] **Step 1: 编写 `mu_argmax_f32` Metal kernel（使用 threadgroup reduction）**
- [ ] **Step 2: 在 `mu_gpu.h` 中声明接口**
- [ ] **Step 3: 集成到 greedy generation 循环**
- [ ] **Step 4: 门禁验证**
- [ ] **Step 5: 提交**

---

## Phase 6: GEMV/Attention Kernel 优化

> 将 naive 顺序循环替换为 SIMD-group 并行归约。
> 设计来源：[mu-performance-optimization.md §2D](file:///Users/will/github/ds4/mineru/docs/mu-performance-optimization.md#L51-L55)。

### Task 6.1: SIMD Group Reduction GEMV

**Files:**
- Modify: [mu_dense.metal](file:///Users/will/github/ds4/mineru/metal/mu_dense.metal)

**动机：** 当前 [mu_dense_probe](file:///Users/will/github/ds4/mineru/metal/mu_dense.metal) 每个线程独立循环 `cols` 次累加。Apple GPU 每个 SIMD-group 有 32 个线程，可以协作并行。

- [ ] **Step 1: 重写 `mu_dense_probe` 使用 `simd_sum` reduction**

每个 threadgroup（32 线程 = 1 SIMD-group）负责一行输出：
```metal
kernel void mu_dense_probe_v2(..., uint tid [[thread_index_in_threadgroup]],
                                   uint tg [[threadgroup_position_in_grid]]) {
    int row = tg;
    float partial = 0.0f;
    for (int c = tid; c < cols; c += 32) {
        partial += x[c] * mu_bf16_to_f32(w[row * cols + c]);
    }
    float sum = simd_sum(partial);
    if (tid == 0) out[row] = sum;
}
```

- [ ] **Step 2: 同样改写 bias 变体和多行变体**
- [ ] **Step 3: 更新 `mu_metal.m` 中的 dispatch 参数（threadgroup size → 32）**
- [ ] **Step 4: 门禁验证（重点检查精度——`simd_sum` 的累加顺序不同于顺序循环）**

> [!WARNING]
> `simd_sum` 改变了浮点累加顺序，可能导致 rounding 差异。必须通过 `--check-trace` 验证 top-1 token 不变。如果精度偏移导致 trace 失败，保留原始 kernel 作为 `_v1` 后缀备选。

- [ ] **Step 5: 提交**

### Task 6.2: Attention Kernel SIMD 优化

**Files:**
- Modify: [mu_attn.metal](file:///Users/will/github/ds4/mineru/metal/mu_attn.metal)

**动机：** 当前 `mu_text_attn_cached`（[mu_attn.metal:180-228](file:///Users/will/github/ds4/mineru/metal/mu_attn.metal#L180-L228)）每个 head 由一个线程处理，64 维 QK 点积循环 3 次。

- [ ] **Step 1: 用 SIMD-group 并行化 head-dim 累加**
- [ ] **Step 2: 门禁验证**
- [ ] **Step 3: 提交**

---

## Phase 7: 端到端验证与报告

### Task 7.1: 10 页 Full-content512 最终验证

来自 [mu-metal-design.md §Benchmark Protocol](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L350-L377)，基准页为：
```text
224, 234, 237, 241, 244, 247, 258, 281, 303, 334
```

- [ ] **Step 1: 运行 10 页 Metal no-fallback benchmark**

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal --max-new-tokens 128 --content-max-new-tokens 512 \
  --timeout 7200 --timing --resume \
  --out /tmp/mu-benchmark-metal-10page-optimized.json \
  --save-output-dir /tmp/mu-optimized-10page
```

- [ ] **Step 2: 与 CPU baseline 和 Transformers/MPS baseline 对比**
- [ ] **Step 3: 更新 [mu-performance-report.md](file:///Users/will/github/ds4/mineru/docs/mu-performance-report.md)**

报告必须包含（来自 [mu-metal-design.md §Benchmark Protocol](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L364-L377)）：
- CPU reference time
- Metal time with fallback disabled
- Transformers/MPS reference time
- block-count accuracy、type accuracy、bbox IoU、content token F1、table cell recall
- Metal fallback count（必须为 0）

### Task 7.2: 精度回归最终确认

- [ ] **Step 1: 确认所有 10 页精度指标达标**

---

## CPU Baseline Reuse Policy

来自 [mu-skills.md §Baseline Reuse Policy](file:///Users/will/github/ds4/mineru/docs/mu-skills.md#L49-L62)：

> CPU 10-page baseline 不应为每次 Metal-only 优化重新运行。除非以下发生变更：
> - CPU 代码
> - 解析语义
> - Token 限制
> - 模型权重
> - 对比逻辑

本计划所有 Phase 均为 Metal-only 优化，不修改 CPU 代码路径，因此复用现有 CPU baseline。

## Safety Rules

- CPU 始终是默认后端和精度参考（[mu-metal-design.md §CPU Reference Policy](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L127-L145)）
- 不要在没有 `--no-cpu-fallback` 的情况下将 Metal 计时视为 Metal 性能（[mu-metal-design.md §Benchmark Protocol](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L299-L303)）
- 每次 C / Objective-C / Metal / Makefile 修改后，先跑 CPU 门禁再跑 Metal 门禁
- 不要把 MinerU Metal 代码合并到 DS4 Metal 代码中（[mu-metal-design.md §Non-Goals](file:///Users/will/github/ds4/mineru/docs/mu-metal-design.md#L33)）
- 不要提交生成的二进制文件或临时日志
- 不要删除或覆盖 worktree 中的用户修改
