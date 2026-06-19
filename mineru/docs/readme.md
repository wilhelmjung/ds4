# ds4.c 架构和原理

> MinerU 相关文档：
> [设计](mu-design.md)，[Metal 化设计](mu-metal-design.md)，[Metal 执行计划](mu-metal-plan.md)，[计划](mu-plan.md)，[性能测试报告](mu-performance-report.md)，[MU Skills](mu-skills.md)。

`ds4.c` 是 DwarfStar 的核心推理引擎。它不是通用 GGUF runner，也不是对其他推理库的包装，而是一条为 DeepSeek V4 Flash / Pro 固化的本地推理管线。

这个文件 deliberately vertical：同一个文件里同时负责 GGUF 加载、模型形状校验、权重绑定、CPU reference kernel、Metal/CUDA 图路径、KV cache、tokenizer、采样、session checkpoint，以及 MTP、SSD streaming、distributed slice 等高级能力。

## 总体定位

`ds4.c` 的基本取舍是用固定模型结构换取性能、确定性和实现可控性。

它只接受已知的 DeepSeek V4 Flash / Pro 形状。模型打开后，代码会严格检查 metadata、tensor 名称、tensor 类型、维度、压缩比例和 RoPE 参数。任何关键字段不符合预期都会立即失败，而不是尝试以通用方式“勉强运行”。

外部程序主要通过 `ds4.h` 使用它：

- `ds4_engine`：已加载模型、后端、权重、词表和全局配置。
- `ds4_session`：一次可变推理时间线，拥有 KV cache、logits、checkpoint 和 speculative state。
- `ds4_session_sync()`：把 session 同步到一个完整 token prefix。
- `ds4_session_eval()`：在当前 checkpoint 后追加一个 token。
- `ds4_session_sample()` / `ds4_session_argmax()`：从当前 logits 中采样。

CLI 和 server 不直接操作 tensor、GPU graph 或 KV 内部结构，这些都封装在 `ds4.c` 内。

## 文件内分层

从上到下，`ds4.c` 大致分为这些部分：

1. DeepSeek V4 Flash / Pro shape profile。
2. GGUF 量化 block 格式和底层 dot kernel。
3. 通用工具：错误处理、分配保护、线程池、cursor 读取。
4. GGUF metadata / tensor directory 解析和 mmap。
5. 固定权重绑定和模型 layout 校验。
6. F16、Q8_0、Q2_K、Q4_K、IQ2_XXS 等 CPU matvec kernel。
7. Hyper-Connection、attention、KV 压缩器、MoE FFN。
8. CPU prefill / decode reference 路径。
9. Metal/CUDA graph prefill / decode 路径。
10. tokenizer、chat prompt、采样。
11. engine/session 公共 API、session snapshot、distributed/MTP 支持。

这不是传统“很多小模块”的结构，而是一个模型专用执行引擎的纵向切片。

## 模型加载

GGUF 加载的核心结构是 `ds4_model`：

```c
typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    ds4_kv *kv;
    ds4_tensor *tensors;
} ds4_model;
```

加载流程：

1. `model_open()` 打开 GGUF 文件。
2. 使用 `mmap()` 映射整个文件。
3. 解析 GGUF header。
4. `parse_metadata()` 记录 metadata key/value 的位置。
5. `parse_tensors()` 记录 tensor 名称、维度、类型和绝对 offset。
6. tensor 数据保持在 mmap 中，不复制到私有权重结构。

CPU 通过 `tensor_data()` 直接访问 mmap 指针。Metal/CUDA 路径则把 mmap 的 tensor slices 映射为 accelerator 侧 buffer/view。

这也是 DwarfStar 能处理几十 GiB 模型文件的关键：加载时不把模型整体复制到堆内存。

## Shape Profile 和严格校验

文件开头定义了两个固定 profile：

- `DS4_SHAPE_FLASH`
- `DS4_SHAPE_PRO`

`config_validate_model()` 从 GGUF metadata 读取模型参数，然后调用 `ds4_select_shape_from_metadata()` 判断当前模型是 Flash 还是 Pro，并设置全局 `g_ds4_shape`。

后续代码通过宏访问当前 shape：

```c
#define DS4_N_LAYER (g_ds4_shape.n_layer)
#define DS4_N_EMBD  (g_ds4_shape.n_embd)
#define DS4_N_EXPERT (g_ds4_shape.n_expert)
```

校验内容包括：

- 层数、hidden size、vocab size。
- attention head 数、KV head 数、head dim。
- LoRA rank、output group 数。
- expert 数、每 token 使用 expert 数、FFN hidden dim。
- Hyper-Connection 数量和 Sinkhorn iteration。
- RoPE 和 YaRN 参数。
- 每层 attention compression ratio。
- SwiGLU clamp 参数。
- 每个 tensor 的类型和维度。

这种严格校验让错误 GGUF 在启动时就失败，避免后续产生难以定位的 logits 偏差。

## 权重绑定

GGUF 中 tensor 以字符串命名。推理热路径不直接字符串查找，而是在 `weights_bind()` 阶段把 tensor 名称绑定成结构体字段。

每层权重用 `ds4_layer_weights` 表示，包含：

- attention HC 控制权重。
- attention norm、Q/KV/out projection。
- attention compressor / indexer compressor。
- FFN HC 控制权重。
- router、routed experts、shared expert。

绑定之后，推理代码直接访问：

```c
layer->attn_q_a
layer->attn_kv
layer->ffn_gate_exps
layer->ffn_down_exps
```

这一步把“字符串化的 GGUF 目录”转换成“DeepSeek V4 专用 pointer table”。

## 一层 Transformer 的执行

每层的逻辑大致是：

```text
HC state
  -> Attention HC pre
  -> Attention RMSNorm
  -> Q projection / KV projection
  -> RoPE
  -> raw + compressed attention
  -> grouped attention output
  -> Attention HC post
  -> FFN HC pre
  -> FFN RMSNorm
  -> routed MoE + shared expert
  -> FFN HC post
```

这里最重要的三个机制是 Hyper-Connection、压缩 KV 和 routed MoE。

## Hyper-Connection

普通 Transformer 通常每个 token 只有一个 residual stream。DeepSeek V4 在这里使用 `DS4_N_HC = 4` 个 Hyper-Connection stream。

每个 attention / FFN 子层前，`hc_pre_from_state_one()` 会：

1. 对完整 HC state 做 RMSNorm。
2. 用小型 F16 projection 生成控制向量。
3. 通过 `hc_split_sinkhorn_one()` 生成三组控制量：
   - pre weights：如何把 4 路 HC state 压成一个普通 hidden vector。
   - post gates：子层输出注入每个 HC stream 的强度。
   - combine matrix：旧 HC streams 如何混合进新 HC streams。

子层后，`hc_post_one()` 把 block output 注回 4 路 HC state。

所以这里不是简单的 residual add，而是学习到的多流 residual mixing。

## Attention 和 RoPE

attention 侧的特点：

- Q 是低秩投影：`attn_q_a -> RMSNorm -> attn_q_b`。
- KV 是单 KV head，宽度为 `DS4_N_HEAD_DIM`。
- Q projection 后会对每个 head 做 RMSNorm。
- RoPE 只作用于每个 head 的 tail 旋转维度。
- 压缩层使用不同的 RoPE base 和 scale。
- attention softmax 包含 learned sink logit。
- attention 输出是 grouped projection：每组 heads 先降到低秩，再投回 embedding width。

attention 不只看最近 token，也会混合长期 compressed KV。

## KV Cache 和压缩 KV

CPU KV cache 的每层状态是 `ds4_layer_cache`：

```c
typedef struct {
    float *raw_kv;
    uint32_t n_raw;
    uint32_t cap_raw;

    uint32_t compress_ratio;
    uint32_t comp_cap;
    uint32_t n_comp;
    float *attn_comp_kv;
    float *attn_state_kv;
    float *attn_state_score;

    uint32_t n_index_comp;
    float *index_comp_kv;
    float *index_state_kv;
    float *index_state_score;
} ds4_layer_cache;
```

KV 分两类：

- raw KV：最近 `DS4_N_SWA` 个 token 的滑动窗口，精确保留。
- compressed KV：长期上下文的压缩摘要。

compression ratio 来自模型 metadata，但必须符合 profile 预期：

- Flash：前 2 层不压缩，后续层交替 ratio 4 / 128。
- Pro：前 2 层 ratio 128，后续层交替 ratio 4 / 128。

压缩器工作方式：

1. 每个 token 通过 compressor projection 得到 KV 和 score。
2. 写入 rolling compression state。
3. 到达 ratio 边界时，对 window 做 per-dimension softmax pooling。
4. pooled row 做 RMSNorm、RoPE 和量化处理。
5. 写入 compressed KV cache。

ratio-4 层还有 indexer cache，用于选择哪些 compressed rows 对当前 token 可见。

## MoE FFN

FFN 分为 shared expert 和 routed experts。

shared expert 是所有 token 都会运行的 Q8_0 SwiGLU MLP：

```text
gate = W_gate_shared x
up   = W_up_shared x
mid  = silu(gate) * up
out  = W_down_shared mid
```

routed MoE 每 token 选择 `DS4_N_EXPERT_USED = 6` 个 expert。

路由有两种：

- 早期 hash layers 使用 `ffn_gate_tid2eid`，按 token id 查表选 expert。
- 后续层用 router projection 得到 expert scores，再 top-k 选择。

routed expert 执行流程：

```text
x -> Q8_K activation
for selected expert:
  gate = IQ2_XXS/Q4_K matvec
  up   = IQ2_XXS/Q4_K matvec
  mid  = silu(gate) * up * router_weight
  mid -> Q8_K activation
  down = Q2_K/Q4_K matvec
sum all selected down outputs
```

这是模型压缩的主要来源：占体积最大的 routed MoE 权重可以使用 2-bit 或 4-bit 格式，而非 routed 权重保持更高精度。

## CPU Reference 路径

CPU 路径主要用于 correctness、诊断和测试，不是主要性能目标。

主要入口：

- `generate_raw_swa_cpu()`
- `prefill_layer_major_cpu()`
- `forward_token_raw_swa_cpu_decode_scratch()`

CPU prefill 是 layer-major：一次处理 prompt token batch，逐层推进。decode 是 token-by-token，使用 `ds4_cpu_decode_scratch` 复用临时 buffer。

`ds4_cpu_decode_scratch` 很重要。decode 热路径避免频繁 malloc/free，一方面提升稳定性，另一方面避免巨大 mmap 与频繁 VM bookkeeping 叠加导致系统层问题。

## GPU Graph 路径

GPU 路径是实际运行主路径。相关函数多以 `metal_graph_*` 命名，但底层抽象也覆盖 CUDA/ROCm 编译路径。

主要入口：

- `generate_metal_graph_raw_swa()`
- `metal_graph_prefill_raw_swa()`
- `metal_graph_prefill_chunked_range()`
- `metal_graph_eval_token_raw_swa()`
- `metal_graph_encode_decode_layer()`
- `metal_graph_encode_layer_batch()`

GPU prefill 和 decode 的策略不同：

```text
prefill:
  layer-major batch
  长 prompt 分 chunk
  每个 chunk 逐层编码 attention + ffn

decode:
  单 token
  embed token
  逐层 decode
  output head
  read logits
```

prefill 使用 layer-major batch 是因为 prompt 阶段 token 多，矩阵乘和 attention 可以充分并行。decode 使用 token-major 是因为生成时每一步依赖上一步 logits 和 KV state。

## Session 和 Checkpoint

`ds4_session` 是推理状态的核心：

```c
struct ds4_session {
    ds4_engine *engine;
    ds4_gpu_graph graph;
    ds4_kv_cache cpu_cache;
    ds4_cpu_decode_scratch cpu_scratch;
    token_vec checkpoint;
    float *logits;
    ...
};
```

最关键的函数是 `ds4_session_sync()`。

上层 CLI / server 可以每次传入完整 prompt token 列表，`ds4_session_sync()` 自动判断如何更新底层状态：

- 如果当前 checkpoint 是新 prompt 的前缀，只扩展 suffix。
- 如果 suffix 很长，用 chunked prefill extension。
- 如果 suffix 很短，用逐 token decode。
- 如果 prompt 与 checkpoint 不匹配，清空状态，从 token 0 重新 prefill。

这样上层 API 可以保持“传完整 transcript”的简单模型，底层仍然复用 KV cache。

## 输出头和采样

最后输出不是直接从单 hidden vector 得到 logits，而是：

```text
final HC state
  -> output HC collapse
  -> output RMSNorm
  -> vocab projection Q8_0
  -> logits
```

采样支持：

- argmax
- temperature
- top-k
- top-p
- min-p
- top logprobs
- 单 token logprob

tokenizer、chat prompt rendering 和 special token 处理也在 `ds4.c` 后段。

## SSD Streaming

SSD streaming 是 Metal 路径的容量模式。

普通路径会尽量让模型权重驻留在 GPU 可寻址内存中。SSD streaming 模式下：

- 非 routed 权重常驻。
- routed MoE experts 按需从 GGUF 文件映射/读取。
- 内存中维护 expert cache。
- 可以通过 hotlist 或 prefill routing 结果预热常用 experts。

这个模式利用现代 SSD 的高吞吐，让“模型是否完全放得进 RAM”从硬门槛变成速度/容量 tradeoff。

## MTP Speculative Decoding

MTP 是可选辅助模型路径。

基本状态机：

1. target model 正常接受一个 token。
2. MTP block 基于当前 frontier draft 一个短 suffix。
3. target graph 批量验证 suffix。
4. 验证通过的前缀被提交。
5. 验证失败时回滚 speculative state，回到普通 decode。

MTP 不是替代主模型采样，target model 仍然定义最终 token stream。

## Distributed Slice

`ds4.c` 支持按层加载模型切片，配合 `ds4_distributed.c` 实现 coordinator / worker。

相关机制包括：

- `load_slice`
- `load_layer_start`
- `load_layer_end`
- optional output head
- layer payload save/restore
- distributed route readiness

在 slice 模式下，`weights_bind()` 只要求本地层范围存在。token embedding 和 output head 根据当前切片角色决定是否必须加载。

## 一条完整推理链

从 CLI 到 token 输出的典型路径：

```text
ds4_cli.c
  -> ds4_engine_open()
      -> ds4_acquire_instance_lock()
      -> model_open()
      -> config_validate_model()
      -> vocab_load()
      -> weights_bind()
      -> ds4_gpu_init()
      -> map model tensor views
  -> ds4_encode_chat_prompt()
  -> ds4_session_create()
  -> ds4_session_sync(prompt)
      -> prefill / resume prefill / decode suffix
  -> loop:
      token = sample(logits)
      emit token text
      ds4_session_eval(token)
```

## 总结

`ds4.c` 是一个 DeepSeek V4 专用、mmap 驱动、KV 压缩感知、MoE 量化优化、GPU graph 优先的本地推理引擎。

它的核心思想是：

- 不做通用 runtime，专注一个模型族。
- 启动时严格校验，避免 silent mismatch。
- 权重留在 mmap 中，减少复制和内存峰值。
- CPU 路径用于 reference 和诊断。
- GPU graph 路径用于实际性能。
- raw SWA + compressed KV 支撑长上下文。
- routed MoE 量化和 SSD streaming 支撑大模型本地运行。
- session checkpoint 把复杂 KV/graph 状态封装在简单 API 后面。
