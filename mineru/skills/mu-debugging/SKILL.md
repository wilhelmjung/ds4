---
name: mu-debugging
description: Use when mu.c, MU Metal, traces, page parsing, benchmark runs, backend parity, fallback detection, or performance measurements fail or behave unexpectedly.
---

# MU Debugging

## Principle

Debug from the smallest reproducible gate upward. Do not optimize or rewrite
until the failure is classified and tied to a stage, artifact, or invariant.

## Triage Order

1. Build failure: `make mu-test` and `make mu`.
2. Trace failure: text trace before layout trace.
3. Backend failure: CPU first, then Metal with `--no-cpu-fallback`.
4. Page failure: single page before sample batch.
5. Accuracy drift: compare CPU-vs-Metal before Transformers-vs-Metal.
6. Performance regression: inspect stage timings before changing kernels.

## Useful Switches

```bash
MU_METAL_DEBUG=1 ./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
MU_TIMING=1 ./mu --backend metal --no-cpu-fallback --image /path/page.png --json
MU_CHECK_TRACE_SCOPE=layout-generation ./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
```

## Failure Classes

| Symptom | First check |
| --- | --- |
| `fallback` appears in stderr | Stage missing or error path returned fallback. |
| CPU passes, Metal trace fails | Stage kernel drift or buffer layout mismatch. |
| Page JSON differs | Compare ordered blocks, bbox, content F1, table cells. |
| Timeout | Check stage timing; isolate page and token limits. |
| Metal slower after optimization | Count host/device copies and buffer allocation churn. |
| PyTorch/MPS comparison regresses | Rerun MPS warm-up and measured passes before blaming Metal. |
| Transformers differs but CPU/Metal match | Check reference DPI, token limits, or parsing expectations. |

## Debug Rules

- Never use fallback-enabled Metal timing as evidence.
- Do not rerun the 10-page CPU baseline for Metal-only changes unless CPU code,
  parsing semantics, token limits, model weights, or comparison logic changed.
- Reuse existing `/tmp/mu-fullcontent512-*` artifacts when the code path has
  not changed.
- Keep page 224 as the first full-content regression target.
- Escalate to the 10-page sample only after page 224 is stable.
- Current Metal is faster than fresh PyTorch/MPS on the local M5 10-page gate;
  use per-kernel Metal timings before proposing new MPS or attention work.

## Evidence To Capture

For any bug fix, record:

- Command and exit code.
- Backend and fallback policy.
- Page number and token limits.
- Relevant stage timings.
- Metrics artifact path.
- Whether CPU, Metal, and Transformers agree.
- For MPS back-to-back results, record both the warm-up and measured artifact
  directories.
