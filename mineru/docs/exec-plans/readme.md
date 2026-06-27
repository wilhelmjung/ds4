# MinerU Execution & Performance Optimization Plans

This directory catalogs the design specifications, roadmaps, and execution plans for the MinerU inference engine and its Metal backend acceleration.

## Active & Outstanding Optimizations

*   **[mu-remaining-optimizations.md](mu-remaining-optimizations.md) [CURRENT]**: The primary checklist for remaining optimization opportunities. Tracks the current status (currently 1.59x faster than PyTorch/MPS) and outlines future plans (Decoder dispatches ICB, Vision FFN fusion, MPSGraph GEMM).

## Completed Performance Plans (June 2026)

*   **[2026-06-23-vision-attention-shape-tuning.md](2026-06-23-vision-attention-shape-tuning.md)**: Shape-sensitive vision attention optimization (resulted in MPSGraph SDPA implementation).
*   **[2026-06-21-resident-decoder-logits.md](2026-06-21-resident-decoder-logits.md)**: Keeps logits and argmax resident in GPU memory to reduce CPU copy roundtrips.
*   **[mu-metal-kernel-optimization-plan.md](mu-metal-kernel-optimization-plan.md)**: Metal kernel optimization specification covering Phase 6 activation fusions and text decoder FFN.
*   **[mu-metal-perf-plan.md](mu-metal-perf-plan.md)**: Performance implementation blueprint for weight caching, command queues, and basic memory management.
*   **[mu-performance-optimization.md](mu-performance-optimization.md)**: Initial diagnostic analysis of bottlenecks and strategic optimization roadmap.

## Foundation Implementation History

*   **[2026-06-16-mu-engine.md](2026-06-16-mu-engine.md)**: Plan for the core MinerU inference loop execution.
*   **[mu-plan.md](mu-plan.md)**: Initial planning document for the C-based MinerU engine.
*   **[mu-metal-plan.md](mu-metal-plan.md)**: Initial design of the Metal backend correctness bridge.
