---
name: mu-design
description: Use when designing or changing the MinerU mu.c engine architecture, backend boundaries, model support scope, public API, or Metal/CPU execution contract.
---

# MU Design

## Principle

`mineru/mu.c` is a model-specific MinerU2.5-Pro engine, not a generic VLM
runtime. Prefer a narrow vertical design with strict validation, CPU reference
correctness, and accelerator paths that are proven against CPU before being
treated as performance data.

## Read First

- `mineru/docs/mu-design.md` for the engine architecture.
- `mineru/docs/mu-metal-design.md` for backend boundaries.
- `mineru/docs/mu-performance-report.md` for current validated behavior.
- `handoff.md` for the current branch state and known gaps.

## Design Rules

- Keep CPU as the default backend and precision reference.
- Keep Metal behind the same public MU API; backend choice is dispatch, not a
  caller-visible tensor API.
- Do not generalize MU into arbitrary Qwen2-VL, safetensors, or VLM support
  unless a concrete model requirement forces it.
- Validate shapes, tensor names, dtypes, and dimensions at load time.
- Keep DS4-specific mechanisms out of MU: MoE, Hyper-Connection, SSD
  streaming, DeepSeek chat rendering, and compressed KV.
- Benchmark claims only count for `--backend metal --no-cpu-fallback`.
- Treat fallback-enabled Metal as a development convenience, never as Metal
  performance evidence.

## Architecture Checklist

Before changing architecture, answer:

- What model invariant does this rely on?
- Which stage owns the behavior: preprocessing, vision, text, generation,
  postprocess, CLI, benchmark, or comparison?
- What CPU reference proves correctness?
- What trace, page smoke, or benchmark will fail if this design is wrong?
- What artifact or report will record the result?

## Preferred Shape

Use stage boundaries:

```text
image preprocess
  -> patch embed / rotary
  -> vision tower
  -> prompt tokenize / multimodal positions
  -> text prefill
  -> cached decode
  -> markup parse
  -> JSON / Markdown result
```

Keep public API and CLI stable unless the user explicitly asks for surface
changes. Add instrumentation behind environment flags or benchmark flags.

## Decision Quality Bar

A design is acceptable only when it names:

- CPU reference path.
- Metal no-fallback behavior.
- Accuracy gate.
- Performance gate.
- Failure mode and diagnostic command.
- Whether CPU baselines should be reused or rerun.
