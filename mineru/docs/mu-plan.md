# mineru/mu.c MinerU2.5-Pro Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mineru/mu.c`, a dedicated MinerU2.5-Pro-2605-1.2B engine that reproduces the local Transformers implementation and grows into a native C/Metal page parser.

**Architecture:** Keep `mineru/mu.c` separate from `ds4.c`: mmap safetensors loader, strict tensor binding, Qwen2-VL processor compatibility, CPU reference inference, MinerU document pipeline, then optional Metal acceleration. Use `/Users/will/github/mineru-model` as the reference implementation and trace source until native output matches it.

**Tech Stack:** C99, Makefile, mmap, safetensors, BF16 weights, float32 reference activations, Accelerate/CBLAS on macOS, Python trace harness, Qwen2-VL tokenizer/image processor compatibility, later Metal kernels.

---

## Current State

Date: 2026-06-16
Branch: `codex/mineru-mu-engine`
Model directory: `/Users/will/github/mineru-model/models`
Reference implementation: `/Users/will/github/mineru-model`

Files already introduced on this branch:

- `mineru/mu.c`: core loader, tokenizer, processor, CPU text path, layout parser, partial multimodal logits path.
- `mineru/mu.h`: public API and test-facing helpers.
- `mineru/mu_cli.c`: `--inspect` and `--check-trace` CLI.
- `mineru/tests/mu_test.c`: C unit tests.
- `mineru/tests/mu_trace.py`: Python trace generator from the local Transformers implementation.
- `mineru/tests/mu_trace_smoke.py`: trace generation smoke test.
- `mineru/tests/mu_cli_smoke.py`: CLI trace smoke test.
- `mineru/tests/mu-traces/.gitignore`: generated trace artifact rules.
- `Makefile`: `mu` and `mu-test` build targets.

Already implemented and previously verified:

- Safetensors mmap loader for the local single-file model.
- Strict model inspection:
  - 681 tensors.
  - BF16 tensor dtype.
  - 24 text layers.
  - hidden size 896.
  - 32 vision layers.
- Qwen2 byte-level BPE tokenizer for current traces.
- Chat prompt rendering for text and layout prompts.
- Image placeholder expansion for layout input.
- Qwen2-VL image preprocessing for the sample layout trace:
  - `image_grid_thw = [1, 74, 74]`
  - `pixel_values_shape = [5476, 1176]`
- M-RoPE position construction for text and layout traces.
- CPU text-only decoder prefill:
  - embedding lookup
  - RMSNorm
  - Q/K/V projection with bias
  - text RoPE
  - grouped-query causal attention
  - output projection
  - SwiGLU MLP
  - final norm
  - tied lm head
- Slow greedy text generation using repeated full-prefill.
- Layout markup parser.
- Python layout trace sidecar for reference image embeddings:
  - `mineru/tests/mu-traces/layout.image_embeds.f32.bin`
  - expected shape `[1369, 896]`

Known passing checkpoints from the last clean verification before the newest layout-logits edits:

```text
trace text chat ok
trace text tokenizer ok
trace text positions ok
trace text logits ok
trace text generation ok
trace layout chat ok
trace layout tokenizer ok
trace layout positions ok
trace layout processor ok
trace layout parser ok
```

Current red checkpoint:

- `mineru/tests/mu_cli_smoke.py` expects `trace layout logits ok`.
- `mineru/mu.c` has `mu_text_top_logits_with_image_embeds()`, but `mineru/mu_cli.c --check-trace` still needs to finish reading the image embedding sidecar and comparing layout logits.
- Rebuild before trusting any status, because recent C edits have not been fully reverified.

## Completion Criteria

Do not consider the MinerU engine complete until all of these are true:

- `make mu-test` passes.
- `make mu` builds the CLI.
- `./mu --check-trace mineru/tests/mu-traces/text.json` passes text chat, tokenizer, positions, logits, and generation checks.
- `./mu --check-trace mineru/tests/mu-traces/layout.json` passes layout chat, tokenizer, processor, positions, parser, and logits checks.
- Native C vision tower can replace the Python-generated image embedding sidecar for layout logits.
- `./mu --image <page-image> --json` emits valid structured JSON.
- `./mu --image <page-image> --markdown` emits non-empty Markdown.
- `/Users/will/github/mineru-model` PDF/page parsing tests pass against the native `mu` path.

## Milestone 0: Restore A Green Trace Baseline

Goal: compile the current branch, finish the in-progress layout-logits trace check, and return the smoke tests to green.

Files:

- Modify: `mineru/mu_cli.c`
- Modify only if compile failures require it: `mineru/mu.c`
- Test: `mineru/tests/mu_cli_smoke.py`

Steps:

- [ ] Build the current C state.

```bash
make mu-test
make mu
```

Expected result: both targets compile. If they fail, fix only the compile/runtime errors caused by the recent M-RoPE and image embedding changes.

- [ ] Keep `position_ids` alive for layout logits.

`mineru/mu_cli.c` currently frees `expected_pos` and `got_pos` before the layout branch. Move the frees after layout logits, or copy `expected_pos` into a separate buffer before freeing. The layout logits call must pass the 3-row M-RoPE positions from the trace.

- [ ] Read layout image embedding sidecar from the trace.

Add the layout branch logic in `mineru/mu_cli.c`:

```c
char *embeds_path = json_get_string(json, "image_embeds_file");
int embeds_shape[2] = {0, 0};
if (!embeds_path ||
    json_get_first_ints(json, "image_embeds_shape", embeds_shape, 2) != 0 ||
    embeds_shape[0] <= 0 || embeds_shape[1] != 896) {
    fprintf(stderr, "trace layout is missing image_embeds_file/image_embeds_shape\n");
    return 1;
}
float *image_embeds = read_float_file(embeds_path, embeds_shape[0] * embeds_shape[1]);
```

Use the existing `read_float_file()` helper if it already compiles. Validate `image_embeds_sample` against the first floats in the file with a tolerance of `1e-4f`.

- [ ] Compare multimodal text logits using sidecar embeddings.

Call:

```c
mu_token_logit got_top[16];
int got_top_n = mu_text_top_logits_with_image_embeds(
    engine,
    got_ids,
    got_n,
    expected_pos,
    image_embeds,
    embeds_shape[0],
    8,
    got_top);
```

Acceptance:

- top-1 token id equals trace top-1 token id.
- at least 6 of the top 8 token ids overlap.
- top-1 logit absolute difference is at most `0.35f`.

When accepted, print:

```text
trace layout logits ok
```

- [ ] Run the baseline verification.

```bash
./mu --check-trace mineru/tests/mu-traces/text.json
./mu --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_cli_smoke.py
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_trace_smoke.py
git diff --check
```

Expected result: all commands pass. This milestone is complete only after `mineru/tests/mu_cli_smoke.py` prints `mu_cli_smoke ok`.

## Milestone 1: Make Sidecar Multimodal Parity Robust

Goal: treat Python image embeddings as a temporary boundary and make text-decoder multimodal scatter deterministic.

Files:

- Modify: `mineru/tests/mu_trace.py`
- Modify: `mineru/mu_cli.c`
- Modify: `mineru/mu.c`
- Test: `mineru/tests/mu_cli_smoke.py`

Steps:

- [ ] Regenerate layout trace from the reference implementation.

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_trace.py \
  --mode layout \
  --device mps \
  --max-new-tokens 64 \
  --out mineru/tests/mu-traces/layout.json
```

Expected result: `mineru/tests/mu-traces/layout.json` references `mineru/tests/mu-traces/layout.image_embeds.f32.bin`, and the binary size is `1369 * 896 * 4` bytes.

- [ ] Verify scatter count and order.

In `mu_text_top_logits_with_image_embeds()`, keep the existing behavior:

- every `151655` token consumes exactly one image embedding row.
- non-image tokens use `model.embed_tokens.weight`.
- return an error if consumed rows do not equal `n_image_embeds`.

Add a CLI error message that prints both counts if they differ.

- [ ] Refactor duplicate decoder code after parity is green.

Keep `mu_text_top_logits_from_embeddings()` as the single decoder body. Change `mu_text_top_logits()` to build token embeddings and call that helper with `position_ids = NULL`. Do this only after Milestone 0 is green, so a regression can be isolated.

- [ ] Verify text-only parity did not change.

```bash
./mu --check-trace mineru/tests/mu-traces/text.json
```

Expected result includes:

```text
trace text logits ok
trace text generation ok
```

## Milestone 2: Implement Native Vision Tower

Goal: remove the Python-generated image embedding sidecar and compute the Qwen2-VL visual embeddings in C.

Reference files:

- `/Users/will/github/mineru-model/.venv/lib/python3.13/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py`
- `/Users/will/github/mineru-model/.venv/lib/python3.13/site-packages/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py`
- `/Users/will/github/mineru-model/models/config.json`
- `/Users/will/github/mineru-model/models/preprocessor_config.json`

Files:

- Modify: `mineru/mu.c`
- Modify: `mineru/mu.h`
- Modify: `mineru/mu_cli.c`
- Modify: `mineru/tests/mu_trace.py`
- Test: `mineru/tests/mu_test.c`
- Test: `mineru/tests/mu_cli_smoke.py`

Steps:

- [ ] Extend the Python trace with internal vision checkpoints.

Add trace fields for:

- patch embedding sample.
- rotary position ids or cos/sin sample.
- first vision block output sample.
- final `visual.merger` output sample.

Keep samples small: first 64 float values per checkpoint.

- [ ] Bind all required vision tensors at load time.

Validate these shapes before inference:

- `visual.patch_embed.proj.weight`
- `visual.blocks.N.norm1.weight`
- `visual.blocks.N.norm1.bias`
- `visual.blocks.N.attn.qkv.weight`
- `visual.blocks.N.attn.qkv.bias`
- `visual.blocks.N.attn.proj.weight`
- `visual.blocks.N.attn.proj.bias`
- `visual.blocks.N.norm2.weight`
- `visual.blocks.N.norm2.bias`
- `visual.blocks.N.mlp.fc1.weight`
- `visual.blocks.N.mlp.fc1.bias`
- `visual.blocks.N.mlp.fc2.weight`
- `visual.blocks.N.mlp.fc2.bias`
- `visual.merger.ln_q.weight`
- `visual.merger.ln_q.bias`
- `visual.merger.mlp.0.weight`
- `visual.merger.mlp.0.bias`
- `visual.merger.mlp.2.weight`
- `visual.merger.mlp.2.bias`

- [ ] Implement patch embedding from `pixel_values`.

Input shape from current trace is `[5476, 1176]`. Output must match Qwen2-VL vision hidden width `1280`.

- [ ] Implement vision transformer blocks.

Use float32 activations and BF16 weights. Match the reference order exactly:

- LayerNorm.
- QKV projection.
- vision RoPE.
- attention.
- projection.
- residual.
- LayerNorm.
- MLP.
- residual.

- [ ] Implement patch merger.

Output must be `[1369, 896]` for the current layout trace. Compare against `image_embeds_sample` before connecting it to the text decoder.

- [ ] Switch layout logits to native vision embeddings.

Keep the sidecar path behind a trace/debug fallback flag while native vision parity is being stabilized. The default `--check-trace mineru/tests/mu-traces/layout.json` should compute embeddings natively once this milestone is complete.

Verification:

```bash
make mu-test
make mu
./mu --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_cli_smoke.py
```

Expected result includes:

```text
trace layout logits ok
```

## Milestone 3: Multimodal Generation And Cache

Goal: generate layout markup from an image prompt, not just compare one-step logits.

Files:

- Modify: `mineru/mu.c`
- Modify: `mineru/mu.h`
- Modify: `mineru/mu_cli.c`
- Modify: `mineru/tests/mu_trace.py`
- Test: `mineru/tests/mu_cli_smoke.py`
- Test: `mineru/tests/mu_test.c`

Steps:

- [ ] Add layout `generated_ids` and `generated_text` checks to `mineru/tests/mu_trace.py`.

Generate with the same max-new-token count and greedy settings as the MinerU Transformers client.

- [ ] Implement multimodal greedy generation.

Initial implementation may repeat full prefill for each generated token, matching the existing text-only slow baseline. It must accept token ids, M-RoPE positions, and image embeddings.

- [ ] Add KV cache after full-prefill generation is correct.

Cache per text decoder layer:

- K/V tensors after M-RoPE.
- current sequence length.
- capacity.
- backend ownership for future Metal support.

- [ ] Verify generated layout parses.

Run:

```bash
./mu --check-trace mineru/tests/mu-traces/layout.json
```

Expected result includes:

```text
trace layout generation ok
trace layout parser ok
```

## Milestone 4: MinerU Page Pipeline

Goal: expose document parsing as the primary product API.

Files:

- Modify: `mineru/mu.c`
- Modify: `mineru/mu.h`
- Modify: `mineru/mu_cli.c`
- Test: `mineru/tests/mu_test.c`
- Add if needed: `mineru/tests/mu_page_smoke.py`

Steps:

- [ ] Implement page image entrypoints.

Public API:

```c
int mu_parse_image_file(mu_engine *e, const char *path, mu_result **out);
int mu_parse_image_rgb(mu_engine *e, const uint8_t *rgb, int width, int height,
                       int stride, mu_result **out);
```

CLI:

```bash
./mu --image /path/to/page.png --json
./mu --image /path/to/page.png --markdown
```

- [ ] Implement layout detection.

Pipeline:

```text
page image
  -> layout prompt
  -> Qwen2-VL processor
  -> vision tower
  -> text decoder generation
  -> layout markup parser
  -> normalized blocks
```

- [ ] Implement block crop, rotate, and resize.

Use normalized bboxes from the layout parser. Keep block image ownership inside `mu_result` so JSON/Markdown writing does not depend on temporary buffers.

- [ ] Implement block recognition prompts.

Map block types to prompts:

- `text`: text recognition.
- `table`: table recognition.
- `formula`: formula recognition.
- `image` and `chart`: image analysis when `image_analysis` is enabled.

- [ ] Emit JSON and Markdown.

JSON must include page size, blocks, bboxes, type, text/content, and angle. Markdown should preserve reading order.

- [ ] Run against the local MinerU samples/tests.

Use the existing scripts in `/Users/will/github/mineru-model` as the acceptance reference. If a script assumes Transformers directly, add a small adapter that shells out to `./mu`.

Verification:

```bash
./mu --image /Users/will/github/mineru-model/sample_page.png --json > /tmp/mu-page.json
./mu --image /Users/will/github/mineru-model/sample_page.png --markdown > /tmp/mu-page.md
/Users/will/github/mineru-model/.venv/bin/python -m json.tool /tmp/mu-page.json >/dev/null
test -s /tmp/mu-page.md
```

## Milestone 5: Metal Acceleration

Goal: make the engine practical on the MacBook after CPU correctness is locked.

Files:

- Modify: `mineru/mu.c`
- Modify: `mineru/mu.h`
- Modify: `mineru/mu_cli.c`
- Modify: `Makefile`
- Add: `metal/mu_dense.metal`
- Add: `metal/mu_norm.metal`
- Add: `metal/mu_attn.metal`
- Add: `metal/mu_vision.metal`

Steps:

- [ ] Add `--backend metal` initialization without changing outputs.

```bash
./mu --inspect --backend metal
```

- [ ] Move dense BF16 matmul to Metal.

Compare CPU and Metal logits on text trace after this step.

- [ ] Move normalization and SwiGLU to Metal.

Compare CPU and Metal logits on text trace after this step.

- [ ] Move text attention to Metal.

Compare CPU and Metal text generation after this step.

- [ ] Move vision tower to Metal.

Compare native vision embeddings and layout logits after this step.

Verification after each kernel group:

```bash
./mu --check-trace mineru/tests/mu-traces/text.json --backend cpu
./mu --check-trace mineru/tests/mu-traces/layout.json --backend cpu
./mu --check-trace mineru/tests/mu-traces/text.json --backend metal
./mu --check-trace mineru/tests/mu-traces/layout.json --backend metal
make mu-test
```

## Risk Register

- Tokenizer and special-token boundaries can invalidate every downstream trace. Keep tokenizer checks first in `--check-trace`.
- M-RoPE indexing differs between text-only and image+text paths. Keep explicit `position_ids` comparison before logits.
- Vision tower CPU execution will be slow, but it is still the clearest correctness boundary before Metal.
- Sidecar image embeddings are a temporary test seam, not the final engine.
- Layout generation can be correct at token level but still fail document usefulness if block postprocess differs from MinerU. Verify both raw layout markup and final JSON/Markdown.
- Do not port DS4 SSD streaming into `mineru/mu.c`; this model is a dense 1.2B Qwen2-VL-derived model, not DS4's large routed MoE profile.

## Always-Run Verification Set

Use this set after any meaningful `mineru/mu.c`, `mineru/mu.h`, `mineru/mu_cli.c`, `mineru/tests/mu_trace.py`, or `Makefile` change:

```bash
make mu-test
make mu
./mu --check-trace mineru/tests/mu-traces/text.json
./mu --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_cli_smoke.py
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_trace_smoke.py
git diff --check
```

If any command fails, keep the failing output with the milestone notes and fix that checkpoint before starting the next milestone.
