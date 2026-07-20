# MinerU Docs Index

This directory keeps MinerU design notes, execution plans, and benchmark reports.

## Start Here

| Document | Purpose |
| --- | --- |
| [architecture/README.md](architecture/README.md) | PlantUML architecture diagrams (system components, pipeline dataflow, Metal engine) |
| [design-docs/mu-design.md](design-docs/mu-design.md) | Dedicated MinerU2.5-Pro engine design |
| [design-docs/mu-architecture.md](design-docs/mu-architecture.md) | Model architecture summary |
| [design-docs/mu-metal-design.md](design-docs/mu-metal-design.md) | Metal backend design |
| [mu-performance-report.md](mu-performance-report.md) | Current benchmark history and conclusions |

## Active Execution Plans

| Document | Status |
| --- | --- |
| [exec-plans/2026-06-21-resident-decoder-logits.md](exec-plans/2026-06-21-resident-decoder-logits.md) | Next task: resident decoder logits |

## Execution Plans

| Document | Purpose |
| --- | --- |
| [exec-plans/2026-06-16-mu-engine.md](exec-plans/2026-06-16-mu-engine.md) | Initial mu engine plan |
| [exec-plans/mu-plan.md](exec-plans/mu-plan.md) | MinerU engine implementation plan |
| [exec-plans/mu-metal-plan.md](exec-plans/mu-metal-plan.md) | Metal backend implementation plan |
| [exec-plans/mu-metal-perf-plan.md](exec-plans/mu-metal-perf-plan.md) | Metal performance execution plan |
| [exec-plans/mu-metal-kernel-optimization-plan.md](exec-plans/mu-metal-kernel-optimization-plan.md) | Kernel optimization plan |
| [exec-plans/mu-performance-optimization.md](exec-plans/mu-performance-optimization.md) | Performance optimization overview |
| [exec-plans/mu-remaining-optimizations.md](exec-plans/mu-remaining-optimizations.md) | Remaining optimization opportunities |

## Reports And Notes

| Document | Purpose |
| --- | --- |
| [mu-phase6-walkthrough.md](mu-phase6-walkthrough.md) | Phase 1-6 walkthrough |
| [mu-skills.md](mu-skills.md) | MU-specific workflow notes |
| [design-docs/ds4-architecture.md](design-docs/ds4-architecture.md) | Archived ds4.c architecture note kept for reference |

## External References

| Reference | Use |
| --- | --- |
| [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf) | Metal 4 / MPP tensor_ops GEMM and fusion guidance |
