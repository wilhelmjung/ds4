---
name: mu-testing
description: Use when validating MinerU mu.c correctness, backend parity, Metal no-fallback behavior, page accuracy, performance reports, or benchmark artifacts.
---

# MU Testing

## Principle

Correctness gates are hierarchical. Unit tests and traces prove mechanics;
page benchmarks prove end-to-end behavior; reports must state exactly which
backend, token limits, pages, and artifacts support each claim.

## Test Matrix

| Level | Purpose |
| --- | --- |
| `make mu-test` | C unit and low-level engine invariants. |
| CPU traces | Precision reference for text/layout mechanics. |
| Metal no-fallback traces | Accelerator stage parity without hidden CPU fallback. |
| Python unit tests | Benchmark and comparison helper correctness. |
| Single-page benchmark | Fast full-content regression, usually page 224. |
| 10-page benchmark | End-to-end smoke corpus accuracy and performance. |

## Commands

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
```

## Full-content512 Checkpoint

Use the 10-page sample:

```text
224, 234, 237, 241, 244, 247, 258, 281, 303, 334
```

Current reference artifacts:

```text
/tmp/mu-mps-b2b-10page-warm-20260624/summary.json
/tmp/mu-mps-b2b-10page-measured-20260624/summary.json
/tmp/mu-mps-b2b-10page-measured-20260624/pages.jsonl
/tmp/mu-metal-layernorm-simd-b2b-10page-20260624.json
/tmp/mu-metal-layernorm-simd-b2b-10page-20260624/metal_page_*.json
/tmp/mu-metal-vs-mps-b2b-10page-20260624.metrics.json
```

Current local M5 benchmark expectation:

- PyTorch/MPS measured: `521.6959s` total, `52.1696s/page`.
- Metal no-fallback: `328.0251s` total, `32.8025s/page`.
- Metal is `1.5904x` faster than PyTorch/MPS on the 10-page gate.
- Metal fallback rows are zero.

## Acceptance Criteria

For pure Metal E2E validation:

- Metal command used `--backend metal --no-cpu-fallback`.
- Completed pages match requested pages.
- Failed pages are zero.
- Fallback rows are zero.
- CPU-vs-Metal has block count exact, ordered type accuracy 1.0, bbox IoU 1.0,
  content token F1 1.0, and table cell recall 1.0.
- Performance report includes exact artifact paths for Metal and any
  PyTorch/MPS comparison.

For PyTorch/MPS back-to-back validation:

- Run one MPS warm-up pass before the measured pass.
- Use `--device mps --batch-size 1 --dpi 120 --max-new-tokens 512`.
- Compare Metal output against the measured MPS `pages.jsonl`.
- Record block count, ordered type accuracy, bbox IoU, content token F1, table
  cell recall, total seconds, mean seconds/page, and relative time.

## Baseline Policy

- CPU is the precision reference.
- Reuse the current 10-page CPU baseline for Metal-only optimization.
- Rerun CPU only when CPU code, parsing semantics, token limits, model weights,
  or comparison logic changed.
- Rerun Metal after each meaningful Metal optimization.
- Rerun MPS only for explicit back-to-back comparisons or when the MPS reference
  environment changed.
- Update `mineru/docs/mu-performance-report.md` only with measured artifact
  paths, not inferred timings.
