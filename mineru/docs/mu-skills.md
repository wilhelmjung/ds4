# MU Skills

Date: 2026-06-19
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

The skills assume the current full-content512 checkpoint:

| Item | Value |
| --- | ---: |
| Sample pages | 10 |
| CPU-vs-Metal exact block-count pages | 10 / 10 |
| CPU-vs-Metal ordered type accuracy | 1.0000 |
| CPU-vs-Metal ordered bbox IoU | 1.0000 |
| CPU-vs-Metal content token F1 | 1.0000 |
| CPU-vs-Metal table cells | 104 / 104 |
| CPU total time | 1381.32s |
| Metal no-fallback total time | 1095.77s |
| PyTorch/MPS (Throttled) total time | 755.93s |
| Metal / CPU speed gap | 1.26x faster |
| Metal / PyTorch MPS speed gap | 1.45x slower |

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

## Documentation Relationship

- `mu-design.md` and `mu-metal-design.md` explain architecture.
- `mu-metal-plan.md` explains the implementation sequence.
- `mu-performance-report.md` records measured results.
- `handoff.md` records current branch state and next work.
- `mineru/skills/*/SKILL.md` records reusable execution behavior for future
  agents working on MU.
