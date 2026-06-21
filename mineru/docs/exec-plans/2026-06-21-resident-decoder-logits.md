# Resident Decoder Logits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce current Metal `text_generate_decode` latency by keeping the final decoder hidden state resident through logits/argmax.

**Architecture:** The current default `layer_resident` decode path runs the 24 decoder layers in one Metal command buffer, then copies the final hidden state back to CPU and calls the existing logits path, which starts another Metal command buffer. The first implementation should only move final norm + vocab projection + argmax into the same resident command buffer and copy back the winning token id/value. Keep the CPU greedy loop and stop-token logic unchanged.

**Tech Stack:** C, Objective-C Metal host code, MSL kernels already present in `mu_sample.metal`, Python source tests, existing `mu_benchmark_pages.py` and `mu_compare_outputs.py`.

---

## Current Evidence

Fresh 10-page full-content512 A/B from 2026-06-21:

| Path | Total s | Mean s/page |
| --- | ---: | ---: |
| PyTorch/Transformers MPS | 841.1548 | 84.1155 |
| Current Metal | 1247.0661 | 124.7066 |

Current Metal mean stage timings:

| Stage | Mean s/page |
| --- | ---: |
| `text_generate_decode` | 51.7080 |
| `layout_vision_encode` | 46.0787 |
| `content_region_vision_encode` | 18.1223 |
| `text_generate_prefill` | 4.4560 |

Decision: do not spend this task on prefill. The cheapest useful resident-decoder cut is resident logits, because the default layer-resident path currently does:

1. Decode layers in one command buffer.
2. `mu_gpu_buf_copy_from(hidden_state, cur_hs_buf, ...)`.
3. `mu_text_top_logits_from_last_hidden(...)`.
4. A separate Metal command buffer for final norm + logits + argmax.

Stop condition: if resident logits improves `text_generate_decode` by less than 5% on a same-run A/B, stop this line and move to `layout_vision_encode`.

## File Structure

- Modify `/Users/will/github/ds4/mineru/mu_gpu.h`
  - Add a `_ctx` logits/argmax API that accepts a resident hidden buffer.
- Modify `/Users/will/github/ds4/mineru/mu_metal.m`
  - Implement `mu_gpu_text_logits_argmax_ctx`.
  - Refactor `mu_gpu_text_logits_argmax` to reuse the `_ctx` helper.
- Modify `/Users/will/github/ds4/mineru/mu.c`
  - Use resident logits in the default `layer_resident` path.
  - Add escape hatch `MU_TEXT_DECODE_NO_RESIDENT_LOGITS=1`.
  - Preserve current hidden-copy logits path as fallback.
- Modify `/Users/will/github/ds4/mineru/tests/test_mu_metal_kernel_sources.py`
  - Add source checks for resident logits wiring and escape hatch.
- Modify `/Users/will/github/ds4/mineru/docs/mu-performance-report.md`
  - Append resident logits A/B results after benchmarking.
- Modify `/Users/will/.gemini/antigravity/brain/d330e4f3-22dd-44e1-bda6-60867e6459c1/task.md`
  - Add completion notes and artifacts.

## Task 1: Add Source Regression Tests

**Files:**
- Modify: `/Users/will/github/ds4/mineru/tests/test_mu_metal_kernel_sources.py`

- [ ] **Step 1: Add the failing source test**

Add this test method to `MuMetalKernelSourceTests`:

```python
    def test_text_decode_resident_logits_path_is_wired(self):
        header = (ROOT / "mineru/mu_gpu.h").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn("mu_gpu_text_logits_argmax_ctx", header)
        self.assertIn("int mu_gpu_text_logits_argmax_ctx", host)
        self.assertIn('getenv("MU_TEXT_DECODE_NO_RESIDENT_LOGITS")', source)
        self.assertIn("text_decode_resident_logits", source)
        self.assertIn("mu_gpu_text_logits_argmax_ctx(ctx", source)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
python -m unittest mineru.tests.test_mu_metal_kernel_sources.MuMetalKernelSourceTests.test_text_decode_resident_logits_path_is_wired
```

Expected: FAIL because `mu_gpu_text_logits_argmax_ctx` and `MU_TEXT_DECODE_NO_RESIDENT_LOGITS` are not wired yet.

## Task 2: Add Resident Logits Host API

**Files:**
- Modify: `/Users/will/github/ds4/mineru/mu_gpu.h`
- Modify: `/Users/will/github/ds4/mineru/mu_metal.m`

- [ ] **Step 1: Add the C API prototype**

Add this prototype near `mu_gpu_text_logits_argmax` in `mu_gpu.h`:

```c
int mu_gpu_text_logits_argmax_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf hidden_state,
                                  mu_gpu_buf final_norm_bf16,
                                  mu_gpu_buf embed_bf16,
                                  float eps, int hidden_dim, int vocab_dim,
                                  mu_gpu_buf out_id, mu_gpu_buf out_val);
```

- [ ] **Step 2: Implement the `_ctx` helper**

Add `mu_gpu_text_logits_argmax_ctx` in `mu_metal.m` before `mu_gpu_text_logits_argmax`.

Implementation shape:

```objc
int mu_gpu_text_logits_argmax_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf hidden_state,
                                  mu_gpu_buf final_norm_bf16,
                                  mu_gpu_buf embed_bf16,
                                  float eps, int hidden_dim, int vocab_dim,
                                  mu_gpu_buf out_id, mu_gpu_buf out_val) {
    if (!ctx || !hidden_state.ptr || !final_norm_bf16.ptr || !embed_bf16.ptr ||
        !out_id.ptr || !out_val.ptr || hidden_dim <= 0 || vocab_dim <= 0 ||
        !ctx->gpu->rmsnorm_bf16_rows || !ctx->gpu->dense_f32_rows ||
        !ctx->gpu->argmax_f32) {
        return -1;
    }

    mu_gpu_buf last = mu_gpu_scratch_alloc_a_ctx(ctx, (unsigned long)hidden_dim * sizeof(float));
    mu_gpu_buf logits = mu_gpu_scratch_alloc_b_ctx(ctx, (unsigned long)vocab_dim * sizeof(float));
    if (!last.ptr || !logits.ptr) return -2;

    id<MTLBuffer> hs_buf = (__bridge id<MTLBuffer>)hidden_state.ptr;
    id<MTLBuffer> norm_buf = (__bridge id<MTLBuffer>)final_norm_bf16.ptr;
    id<MTLBuffer> last_buf = (__bridge id<MTLBuffer>)last.ptr;
    id<MTLBuffer> embed_buf = (__bridge id<MTLBuffer>)embed_bf16.ptr;
    id<MTLBuffer> logits_buf = (__bridge id<MTLBuffer>)logits.ptr;
    id<MTLBuffer> out_id_buf = (__bridge id<MTLBuffer>)out_id.ptr;
    id<MTLBuffer> out_val_buf = (__bridge id<MTLBuffer>)out_val.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->rmsnorm_bf16_rows];
    [ctx->encoder setBuffer:hs_buf offset:hidden_state.offset atIndex:0];
    [ctx->encoder setBuffer:norm_buf offset:final_norm_bf16.offset atIndex:1];
    [ctx->encoder setBuffer:last_buf offset:last.offset atIndex:2];
    [ctx->encoder setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:3];
    [ctx->encoder setBytes:&eps length:sizeof(eps) atIndex:4];
    NSUInteger norm_width = ctx->gpu->rmsnorm_bf16_rows.threadExecutionWidth;
    if (norm_width < 1) norm_width = 1;
    if (norm_width > (NSUInteger)hidden_dim) norm_width = (NSUInteger)hidden_dim;
    [ctx->encoder dispatchThreads:MTLSizeMake((NSUInteger)hidden_dim, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(norm_width, 1, 1)];

    [ctx->encoder setComputePipelineState:ctx->gpu->dense_f32_rows];
    [ctx->encoder setBuffer:last_buf offset:last.offset atIndex:0];
    [ctx->encoder setBuffer:embed_buf offset:embed_bf16.offset atIndex:1];
    [ctx->encoder setBuffer:logits_buf offset:logits.offset atIndex:2];
    [ctx->encoder setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:3];
    [ctx->encoder setBytes:&vocab_dim length:sizeof(vocab_dim) atIndex:4];
    NSUInteger dense_width = ctx->gpu->dense_f32_rows.threadExecutionWidth;
    if (dense_width < 1) dense_width = 1;
    if (dense_width > (NSUInteger)vocab_dim) dense_width = (NSUInteger)vocab_dim;
    [ctx->encoder dispatchThreads:MTLSizeMake((NSUInteger)vocab_dim, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(dense_width, 1, 1)];

    [ctx->encoder setComputePipelineState:ctx->gpu->argmax_f32];
    [ctx->encoder setBuffer:logits_buf offset:logits.offset atIndex:0];
    [ctx->encoder setBuffer:out_id_buf offset:out_id.offset atIndex:1];
    [ctx->encoder setBuffer:out_val_buf offset:out_val.offset atIndex:2];
    [ctx->encoder setBytes:&vocab_dim length:sizeof(vocab_dim) atIndex:3];
    [ctx->encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

    return 0;
}
```

- [ ] **Step 3: Refactor the standalone logits helper**

Keep `mu_gpu_text_logits_argmax(...)` public behavior unchanged. Replace its duplicated encoder body with:

```objc
mu_gpu_cmd_ctx *ctx = NULL;
int rc = mu_gpu_cmd_begin(gpu, &ctx);
if (rc != 0) return rc;
mu_gpu_cmd_set_label(ctx, "text_logits_argmax");
rc = mu_gpu_text_logits_argmax_ctx(ctx, hs, norm_w, embed, eps,
                                   hidden_dim, vocab_dim, out_id_buf_ref,
                                   out_val_buf_ref);
if (rc == 0) rc = mu_gpu_cmd_commit_and_wait(ctx);
else mu_gpu_cmd_discard(ctx);
```

Use local `mu_gpu_buf` wrappers for existing `hs_buf`, `norm_w_buf`, `embed_buf`, `out_id_buf`, and `out_val_buf`. Preserve the final CPU copy of `out_id` and `out_val`.

- [ ] **Step 4: Run source test**

Run:

```bash
python -m unittest mineru.tests.test_mu_metal_kernel_sources.MuMetalKernelSourceTests.test_text_decode_resident_logits_path_is_wired
```

Expected: still FAIL because `mu.c` has not routed the resident path yet.

## Task 3: Wire Default Layer-Resident Decode to Resident Logits

**Files:**
- Modify: `/Users/will/github/ds4/mineru/mu.c`

- [ ] **Step 1: Add the escape hatch and resident logits route**

In `mu_text_cached_step`, inside the default `layer_resident` branch after the 24-layer loop and before `mu_gpu_cmd_commit_and_wait(ctx)`, add a `resident_logits` path guarded by:

```c
int resident_logits = getenv("MU_TEXT_DECODE_NO_RESIDENT_LOGITS") == NULL;
```

Use this logic:

```c
int resident_logits = getenv("MU_TEXT_DECODE_NO_RESIDENT_LOGITS") == NULL;
if (resident_logits) {
    const uint16_t *final_norm = mu_tensor_bf16(e, "model.norm.weight", 1, hidden, 0);
    const uint16_t *embed_w = mu_tensor_bf16(e, "model.embed_tokens.weight", 2, vocab, hidden);
    mu_gpu_buf final_norm_buf = mu_gpu_get_weight_buf(e->gpu, final_norm, hidden * sizeof(unsigned short));
    mu_gpu_buf embed_buf = mu_gpu_get_weight_buf(e->gpu, embed_w, vocab * hidden * sizeof(unsigned short));
    mu_gpu_buf out_id_buf = mu_gpu_scratch_b_at(e->gpu, 0, sizeof(int));
    mu_gpu_buf out_val_buf = mu_gpu_scratch_b_at(e->gpu, sizeof(int), sizeof(float));
    if (final_norm && embed_w && final_norm_buf.ptr && embed_buf.ptr &&
        out_id_buf.ptr && out_val_buf.ptr) {
        rc = mu_gpu_text_logits_argmax_ctx(ctx, cur_hs_buf, final_norm_buf,
                                           embed_buf, eps, hidden, vocab,
                                           out_id_buf, out_val_buf);
        if (rc == 0) {
            rc = mu_gpu_cmd_commit_and_wait(ctx);
            if (rc == 0) {
                int best_id = -1;
                float best_val = -FLT_MAX;
                mu_gpu_buf_copy_from(&best_id, out_id_buf, sizeof(best_id));
                mu_gpu_buf_copy_from(&best_val, out_val_buf, sizeof(best_val));
                out[0].id = best_id;
                out[0].logit = best_val;
                for (int i = 1; i < top_k; i++) {
                    out[i].id = -1;
                    out[i].logit = -FLT_MAX;
                }
                mu_record_metal_stage(e, "text_decode_resident_logits");
                if (timing_stats) {
                    local_timing.cached_logits += mu_time_now_seconds() - logits_start;
                    local_timing.cached_step += mu_time_now_seconds() - step_start;
                    local_timing.steps = 1;
                    mu_text_decode_timing_add(timing_stats, &local_timing);
                }
                free(gate); free(up); free(mid); free(w_tmp);
                return top_k;
            }
        } else {
            mu_gpu_cmd_discard(ctx);
            goto fail;
        }
    }
}
```

If `resident_logits` is disabled or required buffers are unavailable, keep the existing path:

```c
rc = mu_gpu_cmd_commit_and_wait(ctx);
if (rc != 0) goto fail;
mu_gpu_buf_copy_from(hidden_state, cur_hs_buf, hidden * sizeof(float));
int top_rc = mu_text_top_logits_from_last_hidden(e, hidden_state, top_k, out);
```

- [ ] **Step 2: Preserve timing semantics**

Keep `cached_step`, `cached_qkv`, `cached_attn_mlp`, and `cached_logits` populated. `cached_logits` should include resident final norm + dense + argmax plus the tiny id/value copy.

- [ ] **Step 3: Run focused source test**

Run:

```bash
python -m unittest mineru.tests.test_mu_metal_kernel_sources.MuMetalKernelSourceTests.test_text_decode_resident_logits_path_is_wired
```

Expected: PASS.

## Task 4: Correctness Verification

**Files:**
- No new files.

- [ ] **Step 1: Build test binary**

Run:

```bash
make -B mu-test
```

Expected: exit code 0.

- [ ] **Step 2: Run trace checks**

Run:

```bash
MU_TIMING=1 ./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
MU_TIMING=1 ./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
```

Expected: both exit code 0, no fallback rows in stderr.

- [ ] **Step 3: Run Python source and smoke tests**

Run:

```bash
for f in mineru/tests/mu_metal_*.py; do python "$f"; done
python -m unittest mineru.tests.test_mu_metal_kernel_sources mineru.tests.test_mu_text_timing_sources
```

Expected: all tests pass.

## Task 5: Same-Run Decode A/B Benchmark

**Files:**
- No code files.
- Output artifacts under `/tmp`.

- [ ] **Step 1: Run default resident logits on two representative pages**

Run:

```bash
MU_TIMING=1 /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --out /tmp/mu-benchmark-metal-resident-logits-pages224-258.json \
  --pages 224,258 \
  --max-new-tokens 512 \
  --timeout 7200 \
  --keep-going \
  --save-output-dir /tmp/mu-metal-resident-logits-pages224-258 \
  --timing
```

Expected: `completed_pages=2`, `failed_pages=0`, `fallback_rows=0`.

- [ ] **Step 2: Run escape-hatch baseline on the same pages**

Run:

```bash
MU_TEXT_DECODE_NO_RESIDENT_LOGITS=1 MU_TIMING=1 /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --out /tmp/mu-benchmark-metal-no-resident-logits-pages224-258.json \
  --pages 224,258 \
  --max-new-tokens 512 \
  --timeout 7200 \
  --keep-going \
  --save-output-dir /tmp/mu-metal-no-resident-logits-pages224-258 \
  --timing
```

Expected: `completed_pages=2`, `failed_pages=0`, `fallback_rows=0`.

- [ ] **Step 3: Compare outputs**

Run:

```bash
python mineru/tests/mu_compare_outputs.py \
  --pages 224,258 \
  --ref-json-template /tmp/mu-metal-no-resident-logits-pages224-258/metal_page_{page:04d}.json \
  --pred-json-template /tmp/mu-metal-resident-logits-pages224-258/metal_page_{page:04d}.json \
  --out /tmp/mu-resident-logits-pages224-258.metrics.json
```

Expected:

```text
block_count_exact_pages = 2
ordered_type_accuracy = 1.0
mean_content_token_f1 = 1.0
```

- [ ] **Step 4: Compute the speedup**

Run:

```bash
jq -n \
  --slurpfile fast /tmp/mu-benchmark-metal-resident-logits-pages224-258.json \
  --slurpfile base /tmp/mu-benchmark-metal-no-resident-logits-pages224-258.json \
  '{
    fast_total: $fast[0].total_seconds,
    base_total: $base[0].total_seconds,
    total_speedup: ($base[0].total_seconds / $fast[0].total_seconds),
    fast_decode: $fast[0].mean_stage_timings.text_generate_decode,
    base_decode: $base[0].mean_stage_timings.text_generate_decode,
    decode_speedup: ($base[0].mean_stage_timings.text_generate_decode / $fast[0].mean_stage_timings.text_generate_decode),
    fast_logits: $fast[0].mean_stage_timings.text_generate_decode_cached_logits,
    base_logits: $base[0].mean_stage_timings.text_generate_decode_cached_logits
  }'
```

Gate:

- Continue to 10 pages if `decode_speedup >= 1.05`.
- Stop and switch to `layout_vision_encode` if `decode_speedup < 1.05`.

## Task 6: 10-Page Gate If Two-Page A/B Passes

**Files:**
- No code files.
- Output artifacts under `/tmp`.

- [ ] **Step 1: Run 10-page resident logits benchmark**

Run:

```bash
MU_TIMING=1 /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --out /tmp/mu-benchmark-metal-10page-resident-logits.json \
  --pages 224,234,237,241,244,247,258,281,303,334 \
  --max-new-tokens 512 \
  --timeout 7200 \
  --keep-going \
  --save-output-dir /tmp/mu-metal-10page-resident-logits \
  --timing
```

Expected: `completed_pages=10`, `failed_pages=0`, `fallback_rows=0`.

- [ ] **Step 2: Run 10-page escape-hatch benchmark only if hardware state changed**

Use this only if the two-page A/B and the 10-page run are separated by long idle time or thermal state changes.

```bash
MU_TEXT_DECODE_NO_RESIDENT_LOGITS=1 MU_TIMING=1 /Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --out /tmp/mu-benchmark-metal-10page-no-resident-logits.json \
  --pages 224,234,237,241,244,247,258,281,303,334 \
  --max-new-tokens 512 \
  --timeout 7200 \
  --keep-going \
  --save-output-dir /tmp/mu-metal-10page-no-resident-logits \
  --timing
```

Expected: `completed_pages=10`, `failed_pages=0`, `fallback_rows=0`.

- [ ] **Step 3: Compare outputs**

Run:

```bash
python mineru/tests/mu_compare_outputs.py \
  --pages 224,234,237,241,244,247,258,281,303,334 \
  --ref-json-template /tmp/mu-metal-10page-no-resident-logits/metal_page_{page:04d}.json \
  --pred-json-template /tmp/mu-metal-10page-resident-logits/metal_page_{page:04d}.json \
  --out /tmp/mu-resident-logits-10page.metrics.json
```

Expected:

```text
block_count_exact_pages = 10
ordered_type_accuracy = 1.0
mean_content_token_f1 = 1.0
table_exact_cell_recall = 1.0
```

## Task 7: Documentation Update

**Files:**
- Modify: `/Users/will/github/ds4/mineru/docs/mu-performance-report.md`
- Modify: `/Users/will/.gemini/antigravity/brain/d330e4f3-22dd-44e1-bda6-60867e6459c1/task.md`

- [ ] **Step 1: Append performance report section**

Add a section titled:

```markdown
## Resident Decode Logits Checkpoint
```

Include:

- Date.
- Exact commands used.
- Artifact paths.
- Two-page A/B table.
- 10-page table if Task 6 ran.
- Output comparison metrics.
- Decision: keep resident logits default, keep escape hatch, or stop and move to vision encode.

- [ ] **Step 2: Update task file**

Append a compact execution note to `task.md`:

```markdown
## Resident Decode Logits Follow-up

- Status: [completed / stopped after A/B gate]
- Escape hatch: `MU_TEXT_DECODE_NO_RESIDENT_LOGITS=1`
- Main artifact: `/tmp/mu-benchmark-metal-resident-logits-pages224-258.json`
- Result: [decode speedup and correctness metrics]
```

Do not edit the completed text-prefill result tables unless the rerun exposes an actual error.

## Task 8: Final Verification Before Handoff

**Files:**
- No code files.

- [ ] **Step 1: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: exit code 0.

- [ ] **Step 2: Run the minimum test set**

Run:

```bash
make -B mu-test
python -m unittest mineru.tests.test_mu_metal_kernel_sources mineru.tests.test_mu_text_timing_sources
```

Expected: exit code 0.

- [ ] **Step 3: Report concise result**

Report:

- Whether resident logits stayed enabled by default.
- `text_generate_decode` speedup.
- `page_total` speedup.
- Fallback rows.
- Output comparison metrics.
- Next target if speedup is under 5%.

## Execution Choice

Recommended execution mode: inline execution with checkpoints. This change is small and crosses shared C/Metal code, so keeping one thread of context is cheaper than dispatching multiple workers.
