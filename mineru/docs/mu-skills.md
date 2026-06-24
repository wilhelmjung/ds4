# MU Skills

Date: 2026-06-24
Branch: `codex/mineru-metal-backend`

This directory captures the reusable operating practice from the current
`mineru/mu.c` work. These are repository-local Codex-style skills:

```text
mineru/skills/mu-design/SKILL.md
mineru/skills/mu-development/SKILL.md
mineru/skills/mu-debugging/SKILL.md
mineru/skills/mu-testing/SKILL.md
```

They are intentionally project-specific. The goal is to preserve the MU
engineering contract, not to describe generic C, Metal, or VLM development.

## Skill Index

| Skill | Use When | Main Guardrail |
| --- | --- | --- |
| `mu-design` | Changing MU architecture, backend boundaries, model scope, or public API. | Keep MU narrow, model-specific, and CPU-reference-first. |
| `mu-development` | Implementing MU C code, Metal kernels, CLI flags, benchmark helpers, or comparison tools. | Change one stage at a time and verify CPU before Metal. |
| `mu-debugging` | Investigating trace failures, fallback, output drift, timeout, or performance regression. | Reproduce the smallest failing gate before changing code. |
| `mu-testing` | Validating correctness, backend parity, no-fallback Metal, page accuracy, and performance reports. | Only counted Metal benchmarks use `--backend metal --no-cpu-fallback`. |

## Current Baseline Encoded By The Skills

The skills assume the current full-content512 checkpoint on the local Apple
Silicon M5 10-page sample:

| Item | Value |
| --- | ---: |
| Sample pages | 10 |
| Metal no-fallback total time | 328.0251s |
| Metal no-fallback mean time | 32.8025s/page |
| PyTorch/MPS measured total time | 521.6959s |
| PyTorch/MPS measured mean time | 52.1696s/page |
| Metal / PyTorch MPS speed gap | 1.5904x faster |
| Metal fallback rows | 0 |
| Metal-vs-MPS exact block-count pages | 10 / 10 |
| Metal-vs-MPS ordered type accuracy | 1.0000 |
| Metal-vs-MPS ordered bbox IoU | 0.9877 |
| Metal-vs-MPS content token F1 | 1.0000 |
| Metal-vs-MPS table cells | 104 / 104 |
| Reused CPU reference total time | 1381.32s |
| Metal / reused CPU speed gap | 4.2110x faster |

The authoritative performance record is
`mineru/docs/mu-performance-report.md`.

## Baseline Reuse Policy

CPU remains the precision reference, but the full 10-page CPU baseline should
not be rerun for every Metal-only optimization. Reuse the current CPU baseline
unless one of these changed:

- CPU code.
- Parsing semantics.
- Token limits.
- Model weights.
- Comparison logic.

For Metal-only optimization, run page 224 first, then the 10-page Metal
full-content512 sample when the page-level result is stable.

For PyTorch/MPS comparisons, run one warm-up pass and then one measured pass on
the same 10 pages. MPS is now a regression reference, not the immediate
performance blocker.

## Current Optimization Direction

The next local M5 optimization should start with per-kernel timing inside
`text_generate_decode` / `content_region_generate`. The remaining vision FFN
pair is secondary unless decoder dispatch overhead is not cheaply reducible.

## Documentation Relationship

- `mu-design.md` and `mu-metal-design.md` explain architecture.
- `mu-metal-plan.md` explains the implementation sequence.
- `mu-performance-report.md` records measured results.
- `handoff.md` records current branch state and next work.
- `mineru/skills/*/SKILL.md` records reusable execution behavior for future
  agents working on MU.
