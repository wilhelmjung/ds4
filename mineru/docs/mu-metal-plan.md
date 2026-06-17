# mineru/mu.c Metal Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Metal backend for `mineru/mu.c` while keeping the CPU backend as the permanent precision reference.

**Architecture:** Keep public MU APIs stable and dispatch at stage boundaries inside `mineru/mu.c`. Put Metal-specific code under `mineru/` (`mineru/mu_gpu.h`, `mineru/mu_metal.m`, `mineru/metal/*.metal`) and keep CPU tests as the first gate for every milestone. Metal benchmarks only count when CPU fallback is disabled.

**Tech Stack:** C99, Objective-C ARC, Apple Metal, BF16 safetensors mmap weights, Accelerate/CBLAS CPU reference, Python smoke harness from `/Users/will/github/mineru-model/.venv`.

---

## Source Documents

Read these before starting implementation:

- `mineru/docs/mu-metal-design.md`
- `mineru/docs/mu-performance-report.md`
- `mineru/docs/mu-design.md`
- `mineru/docs/mu-plan.md`
- `mineru/mu.h`
- `mineru/mu.c`
- `mineru/mu_cli.c`
- `Makefile`

## Current State

- `mu_backend` already has `MU_BACKEND_CPU` and `MU_BACKEND_METAL` in `mineru/mu.h`.
- `mu_engine_options_default()` currently defaults to CPU in `mineru/mu.c`.
- `mineru/mu_cli.c` currently parses `--model-dir`, `--inspect`, `--check-trace`, `--image`, `--json`, and `--markdown`; it does not parse `--backend`, `--no-cpu-fallback`, or `--compare-backends`.
- `Makefile` builds `mu` from `mineru/mu_cli.o mineru/mu.o` and links CoreGraphics/ImageIO/Accelerate.
- CPU/Accelerate is the correctness path and must remain available.
- Existing trace checks and page smoke tests are the required CPU regression gates.

## File Map

- Modify `mineru/mu.h`: add options and introspection helpers for fallback policy and Metal availability.
- Modify `mineru/mu.c`: add backend dispatch, CPU wrapper functions, fallback counters, and calls into `mu_gpu.h`.
- Modify `mineru/mu_cli.c`: parse backend flags, no-fallback flag, and backend comparison mode.
- Create `mineru/mu_gpu.h`: C interface between `mu.c` and the Metal runtime.
- Create `mineru/mu_metal.m`: Objective-C Metal runtime, pipeline loading, buffer management, and kernel dispatch.
- Create `mineru/metal/mu_dense.metal`: BF16/f32 dense kernels and top-k helpers.
- Create `mineru/metal/mu_norm.metal`: RMSNorm, LayerNorm, residual, SiLU/SwiGLU, rotary helpers.
- Create `mineru/metal/mu_vision.metal`: Qwen2-VL vision tower kernels.
- Create `mineru/metal/mu_attn.metal`: Qwen2 text attention and KV-cache decode kernels.
- Modify `Makefile`: add MinerU Metal object rules without coupling `mu` to DS4 root `metal/*.metal`.
- Create `mineru/tests/mu_backend_smoke.py`: CLI/backend behavior smoke test.
- Create `mineru/tests/mu_compare_backends.py`: page-level CPU/Metal comparison harness.
- Modify `mineru/tests/mu_cli_smoke.py`: keep CPU as the default smoke path and add Metal checks only when available.
- Modify `mineru/docs/mu-performance-report.md`: update after the end-to-end Metal benchmark.

## Required Verification Order

For every implementation task that touches C, Objective-C, Metal, or Makefile files, run CPU gates first:

```bash
make mu-test
make mu
./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_cli_smoke.py
git diff --check
```

Then run Metal gates only on macOS with a Metal device:

```bash
./mu --backend metal --inspect
./mu --backend metal --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --check-trace mineru/tests/mu-traces/layout.json
```

Benchmark gates use fallback disabled:

```bash
./mu --backend metal --no-cpu-fallback --image /path/to/page.png --json >/tmp/mu-metal-page.json
```

## Task 1: Add CLI Backend Contract Tests

**Files:**
- Create: `mineru/tests/mu_backend_smoke.py`
- Modify in Task 2: `mineru/mu_cli.c`
- Test: `mineru/tests/mu_backend_smoke.py`

- [ ] **Step 1: Write the failing backend smoke test**

Create `mineru/tests/mu_backend_smoke.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MU = ROOT / "mu"


def run_mu(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(MU), *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"mu {' '.join(args)} failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def main() -> None:
    cpu = run_mu("--backend", "cpu", "--inspect")
    assert "mu backend=cpu" in cpu.stdout

    bad = run_mu("--backend", "bogus", "--inspect", check=False)
    assert bad.returncode == 2
    assert "--backend must be cpu or metal" in bad.stderr

    no_fallback_cpu = run_mu("--backend", "cpu", "--no-cpu-fallback", "--inspect")
    assert "mu backend=cpu" in no_fallback_cpu.stdout

    metal = run_mu("--backend", "metal", "--inspect", check=False)
    assert metal.returncode in (0, 1)
    if metal.returncode == 0:
        assert "mu backend=metal" in metal.stdout
    else:
        assert "Metal" in metal.stderr or "metal" in metal.stderr

    print("mu_backend_smoke ok")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build current `mu`**

Run:

```bash
make mu
```

Expected: `mu` builds with the current CPU-only implementation.

- [ ] **Step 3: Run the new smoke test and verify it fails**

Run:

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_backend_smoke.py
```

Expected: FAIL because `mineru/mu_cli.c` does not parse `--backend`.

- [ ] **Step 4: Commit the failing test**

```bash
git add mineru/tests/mu_backend_smoke.py
git commit -m "test: add mu backend cli smoke"
```

## Task 2: Add Backend Flags Without Metal Runtime

**Files:**
- Modify: `mineru/mu.h`
- Modify: `mineru/mu.c`
- Modify: `mineru/mu_cli.c`
- Test: `mineru/tests/mu_backend_smoke.py`

- [ ] **Step 1: Extend `mu_engine_options`**

In `mineru/mu.h`, change `mu_engine_options` to:

```c
typedef struct {
    const char *model_dir;
    mu_backend backend;
    int n_threads;
    int max_new_tokens;
    bool inspect_only;
    bool image_analysis;
    bool allow_cpu_fallback;
} mu_engine_options;
```

Add declarations:

```c
bool mu_engine_metal_available(const mu_engine *e);
int mu_engine_cpu_fallback_count(const mu_engine *e);
```

- [ ] **Step 2: Set default fallback policy**

In `mu_engine_options_default()` in `mineru/mu.c`, ensure the returned options include:

```c
opt.backend = MU_BACKEND_CPU;
opt.allow_cpu_fallback = true;
```

- [ ] **Step 3: Add fallback fields to `struct mu_engine`**

In `mineru/mu.c`, extend `struct mu_engine` with:

```c
bool metal_available;
int cpu_fallback_count;
```

- [ ] **Step 4: Add introspection helpers**

In `mineru/mu.c`, add:

```c
bool mu_engine_metal_available(const mu_engine *e) {
    return e && e->metal_available;
}

int mu_engine_cpu_fallback_count(const mu_engine *e) {
    return e ? e->cpu_fallback_count : 0;
}
```

- [ ] **Step 5: Parse `--backend` and `--no-cpu-fallback`**

In `mineru/mu_cli.c`, add parsing branches in `main()`:

```c
        } else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            const char *backend = argv[++i];
            if (!strcmp(backend, "cpu")) {
                opt.backend = MU_BACKEND_CPU;
            } else if (!strcmp(backend, "metal")) {
                opt.backend = MU_BACKEND_METAL;
            } else {
                fprintf(stderr, "--backend must be cpu or metal\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--no-cpu-fallback")) {
            opt.allow_cpu_fallback = false;
```

Update the usage string to include:

```text
[--backend cpu|metal] [--no-cpu-fallback]
```

- [ ] **Step 6: Reject Metal until the runtime exists**

In `mu_engine_open()` in `mineru/mu.c`, after options are copied:

```c
    if (e->opt.backend == MU_BACKEND_METAL) {
        e->metal_available = false;
        if (!e->opt.allow_cpu_fallback) {
            mu_engine_close(e);
            return -20;
        }
        e->cpu_fallback_count++;
        e->opt.backend = MU_BACKEND_CPU;
    }
```

If `mu_engine_open()` currently cannot safely call `mu_engine_close(e)` before initialization is complete, use the existing cleanup path in that function and return `-20`.

- [ ] **Step 7: Print fallback state in summary**

In `mu_engine_summary()`, add:

```c
    fprintf(fp, "mu metal_available=%s cpu_fallback_count=%d allow_cpu_fallback=%s\n",
            e && e->metal_available ? "yes" : "no",
            e ? e->cpu_fallback_count : 0,
            e && e->opt.allow_cpu_fallback ? "yes" : "no");
```

- [ ] **Step 8: Verify backend smoke passes**

Run:

```bash
make mu
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_backend_smoke.py
```

Expected output includes:

```text
mu_backend_smoke ok
```

- [ ] **Step 9: Run CPU regression gates**

Run:

```bash
make mu-test
./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 10: Commit backend flag support**

```bash
git add mineru/mu.h mineru/mu.c mineru/mu_cli.c
git commit -m "feat: add mu backend selection flags"
```

## Task 3: Add Metal Runtime Shell

**Files:**
- Create: `mineru/mu_gpu.h`
- Create: `mineru/mu_metal.m`
- Create: `mineru/metal/mu_dense.metal`
- Create: `mineru/metal/mu_norm.metal`
- Create: `mineru/metal/mu_attn.metal`
- Create: `mineru/metal/mu_vision.metal`
- Modify: `mineru/mu.c`
- Modify: `Makefile`
- Test: `mineru/tests/mu_backend_smoke.py`

- [ ] **Step 1: Create `mineru/mu_gpu.h`**

```c
#ifndef MU_GPU_H
#define MU_GPU_H

#include <stdbool.h>

typedef struct mu_gpu mu_gpu;

int mu_gpu_create(mu_gpu **out);
void mu_gpu_destroy(mu_gpu *gpu);
bool mu_gpu_available(const mu_gpu *gpu);
const char *mu_gpu_device_name(const mu_gpu *gpu);

#endif
```

- [ ] **Step 2: Create `mineru/mu_metal.m`**

```objc
#include "mu_gpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>

struct mu_gpu {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

int mu_gpu_create(mu_gpu **out) {
    if (!out) return -1;
    *out = NULL;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -2;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return -3;
        mu_gpu *gpu = (mu_gpu *)calloc(1, sizeof(*gpu));
        if (!gpu) return -4;
        gpu->device = device;
        gpu->queue = queue;
        *out = gpu;
    }
    return 0;
}

void mu_gpu_destroy(mu_gpu *gpu) {
    if (!gpu) return;
    gpu->queue = nil;
    gpu->device = nil;
    free(gpu);
}

bool mu_gpu_available(const mu_gpu *gpu) {
    return gpu && gpu->device;
}

const char *mu_gpu_device_name(const mu_gpu *gpu) {
    if (!gpu || !gpu->device) return "none";
    return [[gpu->device name] UTF8String];
}
```

- [ ] **Step 3: Create minimal no-op Metal kernel files**

Create `mineru/metal/mu_dense.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void mu_dense_noop(device float *out [[buffer(0)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out[0] = out[0];
}
```

Create `mineru/metal/mu_norm.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void mu_norm_noop(device float *out [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out[0] = out[0];
}
```

Create `mineru/metal/mu_attn.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void mu_attn_noop(device float *out [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out[0] = out[0];
}
```

Create `mineru/metal/mu_vision.metal`:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void mu_vision_noop(device float *out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out[0] = out[0];
}
```

- [ ] **Step 4: Add `mu_gpu *gpu` to `struct mu_engine`**

In `mineru/mu.c`, include `mu_gpu.h` behind a Darwin guard:

```c
#if defined(__APPLE__)
#include "mu_gpu.h"
#endif
```

Extend `struct mu_engine`:

```c
#if defined(__APPLE__)
    mu_gpu *gpu;
#endif
```

- [ ] **Step 5: Create the Metal runtime from `mu_engine_open()`**

In `mu_engine_open()`:

```c
#if defined(__APPLE__)
    if (e->opt.backend == MU_BACKEND_METAL) {
        int gpu_rc = mu_gpu_create(&e->gpu);
        if (gpu_rc == 0 && mu_gpu_available(e->gpu)) {
            e->metal_available = true;
        } else if (!e->opt.allow_cpu_fallback) {
            mu_engine_close(e);
            return -20;
        } else {
            e->metal_available = false;
            e->cpu_fallback_count++;
            e->opt.backend = MU_BACKEND_CPU;
        }
    }
#else
    if (e->opt.backend == MU_BACKEND_METAL) {
        if (!e->opt.allow_cpu_fallback) {
            mu_engine_close(e);
            return -20;
        }
        e->cpu_fallback_count++;
        e->opt.backend = MU_BACKEND_CPU;
    }
#endif
```

Remove the temporary Metal rejection from Task 2 when this code is added.

- [ ] **Step 6: Destroy the Metal runtime**

In `mu_engine_close()`:

```c
#if defined(__APPLE__)
    mu_gpu_destroy(e->gpu);
#endif
```

- [ ] **Step 7: Print Metal device name**

In `mu_engine_summary()`:

```c
#if defined(__APPLE__)
    fprintf(fp, "mu metal_device=%s\n",
            e && e->gpu ? mu_gpu_device_name(e->gpu) : "none");
#else
    fprintf(fp, "mu metal_device=none\n");
#endif
```

- [ ] **Step 8: Update Makefile**

Add near the existing source variables:

```make
MU_METAL_SRCS := $(wildcard mineru/metal/*.metal)
```

On Darwin, add `mu_metal.o` to `mu` and `mu-test` link inputs:

```make
ifeq ($(UNAME_S),Darwin)
MU_OBJS := mineru/mu.o mineru/mu_metal.o
else
MU_OBJS := mineru/mu.o
endif
```

Change Darwin targets:

```make
ifeq ($(UNAME_S),Darwin)
mu: mineru/mu_cli.o $(MU_OBJS) mineru/metal/mu.metallib
	$(CC) $(CFLAGS) -o $@ mineru/mu_cli.o $(MU_OBJS) $(MU_LDLIBS) -framework Metal -framework Foundation

mu-test: mineru/tests/mu_test.o $(MU_OBJS) mineru/metal/mu.metallib
	$(CC) $(CFLAGS) -Imineru -o $@ mineru/tests/mu_test.o $(MU_OBJS) $(MU_LDLIBS) -framework Metal -framework Foundation
	./mu-test
else
mu: mineru/mu_cli.o $(MU_OBJS)
	$(CC) $(CFLAGS) -o $@ mineru/mu_cli.o $(MU_OBJS) $(MU_LDLIBS)

mu-test: mineru/tests/mu_test.o $(MU_OBJS)
	$(CC) $(CFLAGS) -Imineru -o $@ mineru/tests/mu_test.o $(MU_OBJS) $(MU_LDLIBS)
	./mu-test
endif
```

Add object and metallib rules:

```make
mineru/metal/mu.metallib: $(MU_METAL_SRCS)
	xcrun -sdk macosx metal -o mineru/metal/mu.air $(MU_METAL_SRCS)
	xcrun -sdk macosx metallib -o $@ mineru/metal/mu.air

mineru/mu_metal.o: mineru/mu_metal.m mineru/mu_gpu.h $(MU_METAL_SRCS)
	$(CC) $(OBJCFLAGS) -Imineru -c -o $@ mineru/mu_metal.m
```

- [ ] **Step 9: Build and inspect Metal**

Run:

```bash
make mu
./mu --backend metal --inspect
```

Expected on macOS with Metal: command exits 0 and output includes `mu backend=metal` plus a non-`none` `mu metal_device`.

- [ ] **Step 10: Run CPU and backend smoke gates**

Run:

```bash
make mu-test
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_backend_smoke.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 11: Commit Metal runtime shell**

```bash
git add Makefile mineru/mu.h mineru/mu.c mineru/mu_gpu.h mineru/mu_metal.m mineru/metal
git commit -m "feat: add mu metal backend shell"
```

## Task 4: Add Stage-Level CPU Wrappers And Fallback Accounting

**Files:**
- Modify: `mineru/mu.c`
- Modify: `mineru/mu.h`
- Test: `mineru/tests/mu_backend_smoke.py`

- [ ] **Step 1: Rename CPU bodies for dispatch**

In `mineru/mu.c`, rename stage bodies without changing logic:

```c
static int mu_cpu_vision_encode(mu_engine *e, const float *patch_embeds,
                                int rows, int cols,
                                const float *rotary, int rotary_rows, int rotary_cols,
                                float *out, int out_rows, int out_cols);

static int mu_cpu_text_generate_greedy_with_image_embeds(
    mu_engine *e, const int *input_ids, int n_ids,
    int grid_t, int grid_h, int grid_w,
    const float *image_embeds, int n_image_embeds,
    int max_new_tokens, int *out);
```

Keep the public functions named `mu_vision_encode()` and `mu_text_generate_greedy_with_image_embeds()` as wrappers.

- [ ] **Step 2: Add fallback helper**

In `mineru/mu.c`:

```c
static int mu_record_cpu_fallback(mu_engine *e, const char *stage) {
    if (!e) return -1;
    if (e->opt.backend != MU_BACKEND_METAL) return 0;
    if (!e->opt.allow_cpu_fallback) {
        fprintf(stderr, "mu metal stage unavailable without CPU fallback: %s\n",
                stage ? stage : "unknown");
        return -20;
    }
    e->cpu_fallback_count++;
    if (getenv("MU_METAL_DEBUG")) {
        fprintf(stderr, "mu metal fallback to CPU: %s\n",
                stage ? stage : "unknown");
    }
    return 0;
}
```

- [ ] **Step 3: Dispatch `mu_vision_encode()`**

Replace public `mu_vision_encode()` with:

```c
int mu_vision_encode(mu_engine *e, const float *patch_embeds,
                     int rows, int cols,
                     const float *rotary, int rotary_rows, int rotary_cols,
                     float *out, int out_rows, int out_cols) {
    if (e && e->opt.backend == MU_BACKEND_METAL && e->metal_available) {
        int rc = mu_record_cpu_fallback(e, "vision_encode");
        if (rc) return rc;
    }
    return mu_cpu_vision_encode(e, patch_embeds, rows, cols, rotary,
                                rotary_rows, rotary_cols, out, out_rows, out_cols);
}
```

This intentionally falls back until Task 7 connects `mu_gpu_vision_encode()`.

- [ ] **Step 4: Dispatch text generation**

Replace public `mu_text_generate_greedy_with_image_embeds()` with:

```c
int mu_text_generate_greedy_with_image_embeds(mu_engine *e,
                                              const int *input_ids, int n_ids,
                                              int grid_t, int grid_h, int grid_w,
                                              const float *image_embeds,
                                              int n_image_embeds,
                                              int max_new_tokens, int *out) {
    if (e && e->opt.backend == MU_BACKEND_METAL && e->metal_available) {
        int rc = mu_record_cpu_fallback(e, "text_generate");
        if (rc) return rc;
    }
    return mu_cpu_text_generate_greedy_with_image_embeds(
        e, input_ids, n_ids, grid_t, grid_h, grid_w, image_embeds,
        n_image_embeds, max_new_tokens, out);
}
```

- [ ] **Step 5: Verify no-fallback failure**

Run:

```bash
./mu --backend metal --no-cpu-fallback --image /Users/will/github/mineru-model/sample_page.png --json
```

Expected: FAIL with stderr containing `vision_encode` or `text_generate`, because no real Metal compute stage is connected yet.

- [ ] **Step 6: Verify fallback path still works**

Run:

```bash
MU_MAX_NEW_TOKENS=4 ./mu --backend metal --image /Users/will/github/mineru-model/sample_page.png --json >/tmp/mu-metal-fallback.json
/Users/will/github/mineru-model/.venv/bin/python -m json.tool /tmp/mu-metal-fallback.json >/dev/null
```

Expected: both commands exit 0, and `mu_engine_summary()` reports a positive fallback count in inspect mode after a fallback-triggering command is run in process-level tests.

- [ ] **Step 7: Run CPU regression gates**

Run:

```bash
make mu-test
./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 8: Commit fallback dispatch**

```bash
git add mineru/mu.c mineru/mu.h
git commit -m "feat: add mu stage fallback accounting"
```

## Task 5: Implement Metal Dense Kernel Probe

**Files:**
- Modify: `mineru/mu_gpu.h`
- Modify: `mineru/mu_metal.m`
- Modify: `mineru/metal/mu_dense.metal`
- Modify: `mineru/tests/mu_test.c`

- [ ] **Step 1: Add GPU dense probe declaration**

In `mineru/mu_gpu.h`:

```c
int mu_gpu_dense_probe(mu_gpu *gpu,
                       const float *x, const unsigned short *w_bf16,
                       int rows, int cols,
                       float *out);
```

- [ ] **Step 2: Implement BF16 conversion and dense probe kernel**

Replace `mineru/metal/mu_dense.metal` with:

```metal
#include <metal_stdlib>
using namespace metal;

static inline float mu_bf16_to_f32(ushort v) {
    uint bits = ((uint)v) << 16;
    return as_type<float>(bits);
}

kernel void mu_dense_probe(device const float *x [[buffer(0)]],
                           device const ushort *w [[buffer(1)]],
                           device float *out [[buffer(2)]],
                           constant int &cols [[buffer(3)]],
                           uint row [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (int c = 0; c < cols; c++) {
        acc += x[c] * mu_bf16_to_f32(w[row * cols + c]);
    }
    out[row] = acc;
}
```

- [ ] **Step 3: Load the dense pipeline in `mu_metal.m`**

Extend `struct mu_gpu`:

```objc
    id<MTLComputePipelineState> dense_probe;
```

In `mu_gpu_create()`, load the explicit MinerU metallib and pipeline:

```objc
NSError *error = nil;
id<MTLLibrary> library = [device newLibraryWithFile:@"mineru/metal/mu.metallib" error:&error];
if (!library) return -5;
id<MTLFunction> fn = [library newFunctionWithName:@"mu_dense_probe"];
if (!fn) return -6;
gpu->dense_probe = [device newComputePipelineStateWithFunction:fn error:&error];
if (!gpu->dense_probe) return -7;
```

- [ ] **Step 4: Implement `mu_gpu_dense_probe()`**

In `mineru/mu_metal.m`:

```objc
int mu_gpu_dense_probe(mu_gpu *gpu,
                       const float *x, const unsigned short *w_bf16,
                       int rows, int cols,
                       float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->dense_probe) return -1;
    @autoreleasepool {
        id<MTLBuffer> xb = [gpu->device newBufferWithBytes:x
                                                    length:(NSUInteger)cols * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> wb = [gpu->device newBufferWithBytes:w_bf16
                                                    length:(NSUInteger)rows * (NSUInteger)cols * sizeof(unsigned short)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [gpu->device newBufferWithLength:(NSUInteger)rows * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> cb = [gpu->device newBufferWithBytes:&cols
                                                    length:sizeof(int)
                                                   options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cmd = [gpu->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gpu->dense_probe];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:cb offset:0 atIndex:3];
        MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
        NSUInteger tw = MIN((NSUInteger)rows, gpu->dense_probe.maxTotalThreadsPerThreadgroup);
        if (tw == 0) tw = 1;
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -2;
        memcpy(out, [ob contents], (size_t)rows * sizeof(float));
    }
    return 0;
}
```

- [ ] **Step 5: Add a C test guarded by Apple**

In `mineru/tests/mu_test.c`, add:

```c
#if defined(__APPLE__)
#include "mu_gpu.h"
#endif
```

Add a test function:

```c
static int test_mu_gpu_dense_probe(void) {
#if defined(__APPLE__)
    mu_gpu *gpu = NULL;
    if (mu_gpu_create(&gpu) != 0) return 0;
    float x[3] = {1.0f, -2.0f, 0.5f};
    unsigned short w[6] = {
        0x3f80, 0x4000, 0x4040,
        0xbf80, 0x3f80, 0x0000,
    };
    float out[2] = {0};
    int rc = mu_gpu_dense_probe(gpu, x, w, 2, 3, out);
    mu_gpu_destroy(gpu);
    if (rc != 0) return 41;
    if (fabsf(out[0] - (-1.5f)) > 1e-4f) return 42;
    if (fabsf(out[1] - (-3.0f)) > 1e-4f) return 43;
#endif
    return 0;
}
```

Call it from `main()`:

```c
    if ((rc = test_mu_gpu_dense_probe()) != 0) return rc;
```

- [ ] **Step 6: Verify dense probe**

Run:

```bash
make mu-test
```

Expected: `mu_test ok`.

- [ ] **Step 7: Commit dense probe**

```bash
git add mineru/mu_gpu.h mineru/mu_metal.m mineru/metal/mu_dense.metal mineru/tests/mu_test.c Makefile
git commit -m "feat: add mu metal dense probe"
```

## Task 6: Implement Metal Norm And Elementwise Probes

**Files:**
- Modify: `mineru/mu_gpu.h`
- Modify: `mineru/mu_metal.m`
- Modify: `mineru/metal/mu_norm.metal`
- Modify: `mineru/tests/mu_test.c`

- [ ] **Step 1: Add GPU RMSNorm probe declaration**

In `mineru/mu_gpu.h`:

```c
int mu_gpu_rmsnorm_probe(mu_gpu *gpu,
                         const float *x, const float *weight,
                         int n, float eps, float *out);
```

- [ ] **Step 2: Implement RMSNorm kernel**

Replace `mineru/metal/mu_norm.metal` with:

```metal
#include <metal_stdlib>
using namespace metal;

kernel void mu_rmsnorm_probe(device const float *x [[buffer(0)]],
                             device const float *weight [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             constant int &n [[buffer(3)]],
                             constant float &eps [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    float scale = rsqrt(ss / (float)n + eps);
    if ((int)gid < n) {
        out[gid] = x[gid] * scale * weight[gid];
    }
}
```

- [ ] **Step 3: Load and dispatch the RMSNorm pipeline**

In `mineru/mu_metal.m`, add `id<MTLComputePipelineState> rmsnorm_probe;` to `struct mu_gpu`, load `mu_rmsnorm_probe` in `mu_gpu_create()`, and implement `mu_gpu_rmsnorm_probe()` using the same buffer/command pattern as `mu_gpu_dense_probe()`.

Use these buffer indices:

```objc
[enc setBuffer:xb offset:0 atIndex:0];
[enc setBuffer:wb offset:0 atIndex:1];
[enc setBuffer:ob offset:0 atIndex:2];
[enc setBuffer:nb offset:0 atIndex:3];
[enc setBuffer:eb offset:0 atIndex:4];
```

- [ ] **Step 4: Add RMSNorm test**

In `mineru/tests/mu_test.c`, add:

```c
static int test_mu_gpu_rmsnorm_probe(void) {
#if defined(__APPLE__)
    mu_gpu *gpu = NULL;
    if (mu_gpu_create(&gpu) != 0) return 0;
    float x[4] = {1.0f, 2.0f, -3.0f, 4.0f};
    float w[4] = {1.0f, 0.5f, 2.0f, -1.0f};
    float out[4] = {0};
    int rc = mu_gpu_rmsnorm_probe(gpu, x, w, 4, 1e-6f, out);
    mu_gpu_destroy(gpu);
    if (rc != 0) return 51;
    float ss = 1.0f + 4.0f + 9.0f + 16.0f;
    float scale = 1.0f / sqrtf(ss / 4.0f + 1e-6f);
    float exp0 = x[0] * scale * w[0];
    float exp3 = x[3] * scale * w[3];
    if (fabsf(out[0] - exp0) > 1e-4f) return 52;
    if (fabsf(out[3] - exp3) > 1e-4f) return 53;
#endif
    return 0;
}
```

Call it from `main()`.

- [ ] **Step 5: Verify norm probe**

Run:

```bash
make mu-test
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit norm probe**

```bash
git add mineru/mu_gpu.h mineru/mu_metal.m mineru/metal/mu_norm.metal mineru/tests/mu_test.c
git commit -m "feat: add mu metal norm probe"
```

## Task 7: Move Vision Encode To Metal

**Files:**
- Modify: `mineru/mu_gpu.h`
- Modify: `mineru/mu_metal.m`
- Modify: `mineru/metal/mu_dense.metal`
- Modify: `mineru/metal/mu_norm.metal`
- Modify: `mineru/metal/mu_vision.metal`
- Modify: `mineru/mu.c`
- Modify: `mineru/tests/mu_cli_smoke.py`

- [ ] **Step 1: Add GPU vision API**

In `mineru/mu_gpu.h`:

```c
int mu_gpu_vision_encode(mu_gpu *gpu, void *engine,
                         const float *patch_embeds,
                         int rows, int cols,
                         const float *rotary, int rotary_rows, int rotary_cols,
                         float *out, int out_rows, int out_cols);
```

Use `void *engine` so `mu_gpu.h` does not include private `mu_engine` internals.

- [ ] **Step 2: Implement a CPU-equivalent bridge first**

In `mineru/mu_metal.m`, add a temporary bridge that returns `-30`:

```objc
int mu_gpu_vision_encode(mu_gpu *gpu, void *engine,
                         const float *patch_embeds,
                         int rows, int cols,
                         const float *rotary, int rotary_rows, int rotary_cols,
                         float *out, int out_rows, int out_cols) {
    (void)gpu;
    (void)engine;
    (void)patch_embeds;
    (void)rows;
    (void)cols;
    (void)rotary;
    (void)rotary_rows;
    (void)rotary_cols;
    (void)out;
    (void)out_rows;
    (void)out_cols;
    return -30;
}
```

This keeps the compile path explicit before kernels are ported.

- [ ] **Step 3: Connect dispatch to GPU vision**

In `mu_vision_encode()` in `mineru/mu.c`, change the Metal branch to:

```c
#if defined(__APPLE__)
    if (e && e->opt.backend == MU_BACKEND_METAL && e->metal_available) {
        int rc = mu_gpu_vision_encode(e->gpu, e, patch_embeds, rows, cols,
                                      rotary, rotary_rows, rotary_cols,
                                      out, out_rows, out_cols);
        if (rc == 0) return 0;
        rc = mu_record_cpu_fallback(e, "vision_encode");
        if (rc) return rc;
    }
#endif
```

- [ ] **Step 4: Verify fallback still works**

Run:

```bash
MU_METAL_DEBUG=1 MU_MAX_NEW_TOKENS=4 ./mu --backend metal --image /Users/will/github/mineru-model/sample_page.png --json >/tmp/mu-metal-vision-fallback.json
```

Expected: command exits 0 and stderr includes `vision_encode`.

- [ ] **Step 5: Port vision kernels incrementally**

Replace the temporary `-30` implementation with Metal kernels in this order:

```text
patch_embed
rotary application
block0 norm/qkv/attention/output
remaining 31 blocks
spatial merge
projector to hidden size 896
```

After each substage, add or extend a trace comparison in `--check-trace layout.json` before moving to the next substage. Use existing trace probes already checked by `mineru/tests/mu_cli_smoke.py`:

```text
trace layout vision patch ok
trace layout vision rope ok
trace layout vision block0 norm ok
trace layout vision block0 qkv ok
trace layout vision block0 attn ok
trace layout vision block0 output ok
```

- [ ] **Step 6: Add Metal vision smoke to Python test**

In `mineru/tests/mu_cli_smoke.py`, add:

```python
def assert_metal_layout_trace_if_available() -> None:
    result = subprocess.run(
        [
            str(ROOT / "mu"),
            "--backend",
            "metal",
            "--check-trace",
            str(TRACE_DIR / "layout.json"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if result.returncode != 0:
        if "Metal" in result.stderr or "metal" in result.stderr:
            return
        raise AssertionError(
            f"metal layout trace failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert "trace layout logits ok" in result.stdout
```

Call it from `main()` after CPU layout checks.

- [ ] **Step 7: Verify vision parity**

Run:

```bash
make mu-test
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
./mu --backend metal --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_cli_smoke.py
git diff --check
```

Expected: all commands exit 0 on macOS with Metal.

- [ ] **Step 8: Verify no-fallback vision path for layout trace**

Run:

```bash
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
```

Expected: exits 0 once every required layout trace vision stage is Metal-backed.

- [ ] **Step 9: Commit Metal vision encode**

```bash
git add mineru/mu_gpu.h mineru/mu_metal.m mineru/metal/mu_dense.metal mineru/metal/mu_norm.metal mineru/metal/mu_vision.metal mineru/mu.c mineru/tests/mu_cli_smoke.py
git commit -m "feat: run mu vision tower on metal"
```

## Task 8: Move Text Decoder Full Prefill To Metal

**Files:**
- Modify: `mineru/mu_gpu.h`
- Modify: `mineru/mu_metal.m`
- Modify: `mineru/metal/mu_dense.metal`
- Modify: `mineru/metal/mu_norm.metal`
- Modify: `mineru/metal/mu_attn.metal`
- Modify: `mineru/mu.c`
- Modify: `mineru/tests/mu_cli_smoke.py`

- [ ] **Step 1: Add GPU text APIs**

In `mineru/mu_gpu.h`:

```c
int mu_gpu_text_top_logits_with_image_embeds(mu_gpu *gpu, void *engine,
                                             const int *input_ids, int n_ids,
                                             const int *position_ids,
                                             const float *image_embeds,
                                             int n_image_embeds,
                                             int top_k, mu_token_logit *out);

int mu_gpu_text_generate_greedy_with_image_embeds(mu_gpu *gpu, void *engine,
                                                  const int *input_ids, int n_ids,
                                                  int grid_t, int grid_h, int grid_w,
                                                  const float *image_embeds,
                                                  int n_image_embeds,
                                                  int max_new_tokens, int *out);
```

Include `mu.h` in `mu_gpu.h` or forward-declare `mu_token_logit` by including `mu.h`. If including `mu.h` creates a cycle, move `mu_token_logit` to a small shared header `mineru/mu_types.h` and include it from both headers.

- [ ] **Step 2: Add dispatch wrappers**

In `mu_text_top_logits_with_image_embeds()` and `mu_text_generate_greedy_with_image_embeds()` in `mineru/mu.c`, call the GPU function first when backend is Metal. If the GPU function returns `-30`, record CPU fallback and continue through the CPU implementation.

Use this pattern:

```c
#if defined(__APPLE__)
    if (e && e->opt.backend == MU_BACKEND_METAL && e->metal_available) {
        int rc = mu_gpu_text_generate_greedy_with_image_embeds(
            e->gpu, e, input_ids, n_ids, grid_t, grid_h, grid_w,
            image_embeds, n_image_embeds, max_new_tokens, out);
        if (rc == 0) return 0;
        rc = mu_record_cpu_fallback(e, "text_generate");
        if (rc) return rc;
    }
#endif
```

- [ ] **Step 3: Port full-prefill kernels**

Implement Metal kernels for these decoder stages:

```text
token embedding lookup
image embedding scatter
RMSNorm
Q/K/V projections
M-RoPE application
grouped-query causal attention
output projection
SwiGLU MLP
final RMSNorm
lm head top-k
```

Use f32 activations and BF16 weights for the first version.

- [ ] **Step 4: Add logits parity checks**

Extend `--check-trace text.json` and `--check-trace layout.json` output to print backend-specific messages:

```text
trace text logits ok
trace text generation ok
trace layout logits ok
trace layout generation ok
```

The same strings should appear for CPU and Metal so existing tests stay simple.

- [ ] **Step 5: Verify decoder parity**

Run:

```bash
make mu-test
./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
./mu --backend metal --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --check-trace mineru/tests/mu-traces/layout.json
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 6: Verify no-fallback text path**

Run:

```bash
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
```

Expected: both commands exit 0 once Metal covers full-prefill text generation.

- [ ] **Step 7: Commit Metal decoder full prefill**

```bash
git add mineru/mu_gpu.h mineru/mu_metal.m mineru/metal/mu_dense.metal mineru/metal/mu_norm.metal mineru/metal/mu_attn.metal mineru/mu.c mineru/tests/mu_cli_smoke.py
git commit -m "feat: run mu text decoder on metal"
```

## Task 9: Add Metal KV-Cache Decode

**Files:**
- Modify: `mineru/mu_gpu.h`
- Modify: `mineru/mu_metal.m`
- Modify: `mineru/metal/mu_attn.metal`
- Modify: `mineru/mu.c`
- Modify: `mineru/tests/mu_cli_smoke.py`

- [ ] **Step 1: Define GPU decode state**

In `mineru/mu_gpu.h`:

```c
typedef struct mu_gpu_decode mu_gpu_decode;

int mu_gpu_decode_create(mu_gpu *gpu, mu_gpu_decode **out, int max_tokens);
void mu_gpu_decode_destroy(mu_gpu_decode *decode);
```

- [ ] **Step 2: Add per-layer KV buffers**

In `mineru/mu_metal.m`, define:

```objc
struct mu_gpu_decode {
    int max_tokens;
    id<MTLBuffer> key_cache[24];
    id<MTLBuffer> value_cache[24];
};
```

Allocate each layer buffer for 2 KV heads, head dim 64, and `max_tokens` positions. Use the existing model dimensions from `mineru/mu.c`; if private constants are not visible, pass dimensions into `mu_gpu_decode_create()`.

- [ ] **Step 3: Add decode kernels**

In `mineru/metal/mu_attn.metal`, add kernels for:

```text
project current token Q/K/V
append K/V to cache
attention over cached prefix
MLP for current token
lm head top-k
```

Keep full-prefill generation as a fallback comparison path.

- [ ] **Step 4: Add deterministic generation comparison**

In `mineru/tests/mu_cli_smoke.py`, add a check that CPU full-prefill and Metal KV-cache produce the same generated ids for the text trace. Use a CLI mode if one already exists; otherwise add a test-only environment flag:

```text
MU_METAL_DECODE=kv
```

- [ ] **Step 5: Verify KV-cache decode**

Run:

```bash
MU_METAL_DECODE=kv ./mu --backend metal --check-trace mineru/tests/mu-traces/text.json
MU_METAL_DECODE=kv ./mu --backend metal --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_cli_smoke.py
git diff --check
```

Expected: all commands exit 0 and generated tokens match CPU traces.

- [ ] **Step 6: Commit KV-cache decode**

```bash
git add mineru/mu_gpu.h mineru/mu_metal.m mineru/metal/mu_attn.metal mineru/mu.c mineru/tests/mu_cli_smoke.py
git commit -m "feat: add mu metal kv decode"
```

## Task 10: Add Backend Comparison Harness

**Files:**
- Create: `mineru/tests/mu_compare_backends.py`
- Modify: `mineru/mu_cli.c`
- Test: `mineru/tests/mu_compare_backends.py`

- [ ] **Step 1: Add comparison script**

Create `mineru/tests/mu_compare_backends.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PDF = Path("/Users/will/github/mineru-model/testdata/nasa_systems_engineering_handbook_rev2.pdf")
MU = ROOT / "mu"


def render_page(page_number: int, out_path: Path) -> None:
    doc = fitz.open(PDF)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
    Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(out_path)


def run_backend(backend: str, image: Path) -> list[dict]:
    preflight = [str(MU), "--backend", backend, "--inspect"]
    if backend == "metal":
        preflight.insert(3, "--no-cpu-fallback")
    result = subprocess.run(
        preflight,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if backend == "metal" and result.returncode != 0:
        raise RuntimeError(f"metal unavailable or fallback path incomplete:\n{result.stderr}")

    cmd = [str(MU), "--backend", backend, "--image", str(image), "--json"]
    if backend == "metal":
        cmd.insert(3, "--no-cpu-fallback")
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    return json.loads(result.stdout)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        image = Path(td) / "page224.png"
        render_page(224, image)
        cpu = run_backend("cpu", image)
        metal = run_backend("metal", image)
    assert [b["type"] for b in cpu] == [b["type"] for b in metal]
    assert len(cpu) == len(metal)
    for a, b in zip(cpu, metal):
        assert a["type"] == b["type"]
        assert a.get("content") == b.get("content")
    print("mu_compare_backends ok")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify comparison script failure mode before full Metal completion**

Run:

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_compare_backends.py
```

Expected before full Metal end-to-end is complete: FAIL with a clear Metal unavailability or no-fallback message. Confirm the final Metal page command printed in logs or debugger form is:

```bash
./mu --backend metal --no-cpu-fallback --image PAGE --json
```

- [ ] **Step 3: Add `--compare-backends` CLI mode**

In `mineru/mu_cli.c`, add a boolean:

```c
int compare_backends = 0;
```

Parse:

```c
        } else if (!strcmp(argv[i], "--compare-backends")) {
            compare_backends = 1;
```

Require `--image PATH` with `--compare-backends`.

Implement comparison by opening one CPU engine and one Metal engine with `allow_cpu_fallback = false`, running `mu_parse_image_file()` for both, and printing a compact JSON summary:

```json
{"block_count_equal":true,"type_equal":true,"content_equal":true}
```

- [ ] **Step 4: Verify comparison mode**

Run:

```bash
./mu --compare-backends --image /Users/will/github/mineru-model/sample_page.png
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_compare_backends.py
```

Expected after full Metal end-to-end is complete: both commands exit 0.

- [ ] **Step 5: Commit comparison harness**

```bash
git add mineru/mu_cli.c mineru/tests/mu_compare_backends.py
git commit -m "test: compare mu cpu and metal backends"
```

## Task 11: Run 10-Page Metal Benchmark

**Files:**
- Create: `mineru/tests/mu_benchmark_pages.py`
- Modify: `mineru/docs/mu-performance-report.md`

- [ ] **Step 1: Create benchmark script**

Create `mineru/tests/mu_benchmark_pages.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MU = ROOT / "mu"
PDF = Path("/Users/will/github/mineru-model/testdata/nasa_systems_engineering_handbook_rev2.pdf")
PAGES = [224, 234, 237, 241, 244, 247, 258, 281, 303, 334]


def render_page(page_number: int, out_path: Path) -> None:
    doc = fitz.open(PDF)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72), alpha=False)
    Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(out_path)


def run_one(backend: str, image: Path) -> tuple[float, list[dict]]:
    cmd = [str(MU), "--backend", backend, "--image", str(image), "--json"]
    if backend == "metal":
        cmd.insert(3, "--no-cpu-fallback")
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1200,
    )
    elapsed = time.perf_counter() - start
    return elapsed, json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "metal"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for page in PAGES:
            image = tmp / f"page_{page:04d}.png"
            render_page(page, image)
            seconds, blocks = run_one(args.backend, image)
            rows.append(
                {
                    "page": page,
                    "seconds": seconds,
                    "blocks": len(blocks),
                    "types": [b.get("type") for b in blocks],
                }
            )
            print(json.dumps(rows[-1]), flush=True)

    total = sum(r["seconds"] for r in rows)
    output = {
        "backend": args.backend,
        "pages": PAGES,
        "total_seconds": total,
        "mean_seconds": total / len(rows),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run CPU reference benchmark**

Run:

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend cpu \
  --out /tmp/mu-benchmark-cpu.json
```

Expected: writes `/tmp/mu-benchmark-cpu.json` with 10 rows.

- [ ] **Step 3: Run Metal benchmark**

Run:

```bash
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_benchmark_pages.py \
  --backend metal \
  --out /tmp/mu-benchmark-metal.json
```

Expected: writes `/tmp/mu-benchmark-metal.json` with 10 rows and zero CPU fallback because the command uses `--no-cpu-fallback`.

- [ ] **Step 4: Update performance report**

Generate the Markdown section from measured JSON:

```bash
python3 - <<'PY' >/tmp/mu-metal-benchmark-section.md
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

cpu = json.loads(Path("/tmp/mu-benchmark-cpu.json").read_text())
metal = json.loads(Path("/tmp/mu-benchmark-metal.json").read_text())

print("## Metal Backend Benchmark")
print()
print(f"Date: {date.today().isoformat()}")
print()
print("| Backend | Total s | Mean s/page | CPU fallback |")
print("| --- | ---: | ---: | ---: |")
print(f"| CPU reference | {cpu['total_seconds']:.2f} | {cpu['mean_seconds']:.2f} | 0 |")
print(f"| Metal | {metal['total_seconds']:.2f} | {metal['mean_seconds']:.2f} | 0 |")
print("| Transformers/MPS | 386.98 | 38.70 | n/a |")
print()
print("The Metal benchmark used `--backend metal --no-cpu-fallback` on the same sampled")
print("pages as the baseline report.")
PY
```

Append `/tmp/mu-metal-benchmark-section.md` to `mineru/docs/mu-performance-report.md` after the current conclusion section, keeping a blank line before the new heading.

- [ ] **Step 5: Verify report and benchmark scripts**

Run:

```bash
/Users/will/github/mineru-model/.venv/bin/python -m json.tool /tmp/mu-benchmark-cpu.json >/dev/null
/Users/will/github/mineru-model/.venv/bin/python -m json.tool /tmp/mu-benchmark-metal.json >/dev/null
rg -n "Metal Backend Benchmark|--no-cpu-fallback" mineru/docs/mu-performance-report.md
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit benchmark update**

```bash
git add mineru/tests/mu_benchmark_pages.py mineru/docs/mu-performance-report.md
git commit -m "perf: benchmark mu metal backend"
```

## Final Verification Set

Run this after Task 11:

```bash
make mu-test
make mu
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_backend_smoke.py
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_cli_smoke.py
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_trace_smoke.py
./mu --backend cpu --check-trace mineru/tests/mu-traces/text.json
./mu --backend cpu --check-trace mineru/tests/mu-traces/layout.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/text.json
./mu --backend metal --no-cpu-fallback --check-trace mineru/tests/mu-traces/layout.json
/Users/will/github/mineru-model/.venv/bin/python mineru/tests/mu_compare_backends.py
git diff --check
```

Expected:

```text
mu_test ok
mu_backend_smoke ok
mu_cli_smoke ok
mu_trace_smoke ok
```

The Metal trace commands must exit 0 with no CPU fallback in benchmark mode.

## Completion Criteria

- CPU backend remains the default and all CPU trace checks pass.
- Metal backend initializes through `mineru/mu_metal.m`.
- `--backend metal --no-cpu-fallback` runs traces without CPU fallback.
- `--compare-backends` confirms page-level CPU/Metal equality on smoke pages.
- 10-page benchmark report is updated with Metal timing and zero fallback count.
- DS4 root Metal files are not modified for MinerU-specific kernels.
- `mineru/docs/mu-performance-report.md` clearly separates CPU reference, Metal, and Transformers/MPS numbers.
