---
name: mu-development
description: Use when implementing or modifying MinerU mu.c source, Metal kernels, CLI flags, benchmark helpers, comparison tools, or MU documentation-backed behavior.
---

# MU Development

## Principle

Move one stage at a time, keep CPU behavior stable, and make every Metal change
observable through traces, smoke tests, or benchmark artifacts.

## Read First

- `mineru/docs/mu-metal-plan.md` for the current implementation sequence.
- `mineru/docs/mu-performance-report.md` for the latest baseline.
- Relevant source: `mineru/mu.c`, `mineru/mu_metal.m`, `mineru/mu_gpu.h`,
  `mineru/metal/*.metal`, `mineru/tests/*.py`.

## Workflow

1. Inspect current state with `git status --short --branch`.
2. Identify the smallest stage being changed.
3. Add or update a focused test first when behavior changes.
4. Implement using existing stage dispatch patterns.
5. Keep CPU default unchanged.
6. Keep fallback accounting correct.
7. Verify CPU before Metal.
8. Update docs only after measured evidence exists.

## Current Performance Direction

Current native Metal sequential is faster than the existing PyTorch/MPS
warm-rerun 10-page baseline. With Vision dense 2SG, QKV 2SG, BF16 KV cache,
and decode ICB enabled, the refreshed Metal artifact is:

```text
/tmp/mu_10page_seq_qkv2sg_refresh.json
```

It completed 10 / 10 pages with zero fallback rows in `174.984202s`
(`17.498420s/page`), or `4.07x` faster than the existing PyTorch/MPS warm-rerun
reference. Treat MPS as a regression reference, not the immediate blocker.

Start the next optimization cycle from fresh `--timing` / split-profile
evidence. Do not spend another cycle on LayerNorm, attention, or 1SG-vs-2SG
micro-variants unless the profile changes. Keep `MU_DENSE_ROWS_NO_2SG=1` as the
A/B escape hatch for the Vision dense and QKV 2SG defaults.

## Coding Rules

- Use `apply_patch` for manual edits.
- Do not commit generated artifacts: `mu`, `mu-test`, `*.o`, benchmark logs.
- Prefer existing helpers and stage names over new abstractions.
- Add comments only for non-obvious stage or memory ownership rules.
- For Metal, keep `--no-cpu-fallback` meaningful; missing stages must fail
  instead of silently falling back.
- For performance work, reduce copies or dispatch cost only after preserving
  exact CPU parity.

## Common Commands

```bash
make mu-test
make mu
./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python -m unittest \
  mineru.tests.test_mu_benchmark_pages \
  mineru.tests.test_mu_compare_outputs
git diff --check
```

## Commit Bar

Before committing:

- Build/test commands above have fresh output.
- `git status` contains only intended tracked changes plus known local
  generated artifacts.
- Reports distinguish CPU, Metal no-fallback, and Transformers/MPS.
- Any performance number names the exact artifact path.
- Skill docs are updated when the accepted baseline or default optimization
  direction changes.
