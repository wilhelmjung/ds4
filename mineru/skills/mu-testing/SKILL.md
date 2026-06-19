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
/tmp/mu-benchmark-cpu-page224-fullcontent512-kvcache-baseline.json
/tmp/mu-benchmark-cpu-9remaining-fullcontent512-kvcache-baseline.json
/tmp/mu-benchmark-metal-page224-fullcontent512-kvcache.json
/tmp/mu-benchmark-metal-9remaining-fullcontent512-kvcache.json
/tmp/mu-fullcontent512-kvcache-10-combined/cpu-vs-metal.metrics.json
/tmp/mu-fullcontent512-kvcache-10-combined/transformers120-vs-metal.metrics.json
```

## Acceptance Criteria

For pure Metal E2E validation:

- Metal command used `--backend metal --no-cpu-fallback`.
- Completed pages match requested pages.
- Failed pages are zero.
- Fallback rows are zero.
- CPU-vs-Metal has block count exact, ordered type accuracy 1.0, bbox IoU 1.0,
  content token F1 1.0, and table cell recall 1.0.
- Performance report includes CPU, Metal, and Transformers/MPS timing.

## Baseline Policy

- CPU is the precision reference.
- Reuse the current 10-page CPU baseline for Metal-only optimization.
- Rerun CPU only when CPU code, parsing semantics, token limits, model weights,
  or comparison logic changed.
- Rerun Metal after each meaningful Metal optimization.
- Update `mineru/docs/mu-performance-report.md` only with measured artifact
  paths, not inferred timings.
