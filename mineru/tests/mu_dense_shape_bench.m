#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "mu_gpu.h"

#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int cols;
    int out_cols;
} bench_shape;

static double now_ms(void) {
    static mach_timebase_info_data_t info;
    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t t = mach_absolute_time();
    return (double)t * (double)info.numer / (double)info.denom / 1000000.0;
}

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}

static float round_bf16(float x) {
    return bf16_to_f32(f32_to_bf16(x));
}

static uint32_t rng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float random_small(uint32_t *state) {
    uint32_t v = rng_next(state);
    float u = (float)(v & 0x00ffffffu) / 16777216.0f;
    return (u - 0.5f) * 0.125f;
}

static void fill_inputs(float *x, uint16_t *w_bf16, uint16_t *bias_bf16,
                        int rows, int cols, int out_cols) {
    uint32_t state = 0x12345678u ^ (uint32_t)rows ^ ((uint32_t)cols << 8) ^
                     ((uint32_t)out_cols << 16);
    for (size_t i = 0; i < (size_t)rows * (size_t)cols; i++) {
        x[i] = round_bf16(random_small(&state));
    }
    for (size_t i = 0; i < (size_t)out_cols * (size_t)cols; i++) {
        w_bf16[i] = f32_to_bf16(random_small(&state));
    }
    for (int i = 0; i < out_cols; i++) bias_bf16[i] = 0;
}

static void bf16_to_f32_array(const uint16_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = bf16_to_f32(src[i]);
}

static double max_abs_diff(const float *a, const float *b, size_t n) {
    double max_diff = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

static double checksum(const float *x, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i += 97) sum += x[i];
    return sum;
}

static double bench_cpu_sgemm(const float *x, const float *w_f32, float *out,
                              int rows, int cols, int out_cols,
                              int warmup, int iters) {
    for (int i = 0; i < warmup; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, out_cols, cols, 1.0f,
                    x, cols, w_f32, cols, 0.0f, out, out_cols);
    }
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, out_cols, cols, 1.0f,
                    x, cols, w_f32, cols, 0.0f, out, out_cols);
    }
    return (now_ms() - t0) / (double)iters;
}

static int run_mps_batch(id<MTLCommandQueue> queue,
                         MPSMatrixMultiplication *kernel,
                         MPSMatrix *x_matrix,
                         MPSMatrix *w_matrix,
                         MPSMatrix *out_matrix,
                         int repeats) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) return -1;
    for (int i = 0; i < repeats; i++) {
        [kernel encodeToCommandBuffer:cb
                            leftMatrix:x_matrix
                           rightMatrix:w_matrix
                          resultMatrix:out_matrix];
    }
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted ? 0 : -2;
}

static double bench_mps(id<MTLDevice> device, const float *x, const float *w_f32,
                        float *out, int rows, int cols, int out_cols,
                        int warmup, int iters) {
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return -1.0;

    NSUInteger x_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
    NSUInteger w_bytes = (NSUInteger)out_cols * (NSUInteger)cols * sizeof(float);
    NSUInteger out_bytes = (NSUInteger)rows * (NSUInteger)out_cols * sizeof(float);
    id<MTLBuffer> x_buf = [device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> w_buf = [device newBufferWithBytes:w_f32 length:w_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
    if (!x_buf || !w_buf || !out_buf) return -2.0;

    MPSMatrixDescriptor *x_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                              columns:(NSUInteger)cols
                                             rowBytes:(NSUInteger)cols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *w_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)out_cols
                                              columns:(NSUInteger)cols
                                             rowBytes:(NSUInteger)cols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *out_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                              columns:(NSUInteger)out_cols
                                             rowBytes:(NSUInteger)out_cols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrix *x_matrix = [[MPSMatrix alloc] initWithBuffer:x_buf descriptor:x_desc];
    MPSMatrix *w_matrix = [[MPSMatrix alloc] initWithBuffer:w_buf descriptor:w_desc];
    MPSMatrix *out_matrix = [[MPSMatrix alloc] initWithBuffer:out_buf descriptor:out_desc];
    MPSMatrixMultiplication *kernel =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                          transposeLeft:NO
                                         transposeRight:YES
                                             resultRows:(NSUInteger)rows
                                          resultColumns:(NSUInteger)out_cols
                                        interiorColumns:(NSUInteger)cols
                                                  alpha:1.0
                                                   beta:0.0];
    if (!x_matrix || !w_matrix || !out_matrix || !kernel) return -3.0;

    if (run_mps_batch(queue, kernel, x_matrix, w_matrix, out_matrix, warmup) != 0) return -4.0;
    double t0 = now_ms();
    if (run_mps_batch(queue, kernel, x_matrix, w_matrix, out_matrix, iters) != 0) return -5.0;
    double ms = (now_ms() - t0) / (double)iters;
    memcpy(out, [out_buf contents], out_bytes);
    return ms;
}

static int run_metal_current_batch(mu_gpu *gpu, const float *x,
                                   const uint16_t *w_bf16,
                                   const uint16_t *bias_bf16,
                                   int rows, int cols, int out_cols,
                                   int repeats) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;

    unsigned long x_bytes = (unsigned long)rows * (unsigned long)cols * sizeof(float);
    unsigned long w_bytes = (unsigned long)out_cols * (unsigned long)cols * sizeof(uint16_t);
    unsigned long bias_bytes = (unsigned long)out_cols * sizeof(uint16_t);
    unsigned long out_bytes = (unsigned long)rows * (unsigned long)out_cols * sizeof(float);
    mu_gpu_buf x_buf = mu_gpu_scratch_alloc_a_ctx(ctx, x_bytes);
    mu_gpu_buf w_buf = mu_gpu_get_weight_buf(gpu, w_bf16, w_bytes);
    mu_gpu_buf bias_buf = mu_gpu_get_weight_buf(gpu, bias_bf16, bias_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!x_buf.ptr || !w_buf.ptr || !bias_buf.ptr || !out_buf.ptr) {
        mu_gpu_cmd_discard(ctx);
        return -10;
    }
    mu_gpu_buf_copy_to(x_buf, x, x_bytes);
    for (int i = 0; i < repeats; i++) {
        rc = mu_gpu_dense_bf16_bias_rows_ctx(ctx, x_buf, w_buf, bias_buf,
                                             rows, cols, out_cols, out_buf);
        if (rc != 0) {
            mu_gpu_cmd_discard(ctx);
            return rc;
        }
    }
    return mu_gpu_cmd_commit_and_wait(ctx);
}

static double bench_metal_current(mu_gpu *gpu, const float *x,
                                  const uint16_t *w_bf16,
                                  const uint16_t *bias_bf16,
                                  int rows, int cols, int out_cols,
                                  int warmup, int iters) {
    if (run_metal_current_batch(gpu, x, w_bf16, bias_bf16,
                                rows, cols, out_cols, warmup) != 0) {
        return -1.0;
    }
    double t0 = now_ms();
    if (run_metal_current_batch(gpu, x, w_bf16, bias_bf16,
                                rows, cols, out_cols, iters) != 0) {
        return -2.0;
    }
    return (now_ms() - t0) / (double)iters;
}

static double bench_layernorm(mu_gpu *gpu, const float *x, const uint16_t *w,
                              const uint16_t *b, float *out,
                              int rows, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) {
        if (mu_gpu_layernorm_bf16_rows(gpu, x, w, b, rows, 1280, 1e-6f, out) != 0) return -1.0;
    }
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        if (mu_gpu_layernorm_bf16_rows(gpu, x, w, b, rows, 1280, 1e-6f, out) != 0) return -2.0;
    }
    return (now_ms() - t0) / (double)iters;
}

static double bench_attn(mu_gpu *gpu, const float *q, const float *kv,
                         const float *rotary, float *out,
                         int rows, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) {
        if (mu_gpu_vision_attn_rows(gpu, q, kv, rotary, rows, out) != 0) return -1.0;
    }
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        if (mu_gpu_vision_attn_rows(gpu, q, kv, rotary, rows, out) != 0) return -2.0;
    }
    return (now_ms() - t0) / (double)iters;
}

static double bench_gelu(mu_gpu *gpu, const float *x, float *out,
                         int n, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) {
        if (mu_gpu_vision_quick_gelu_bf16(gpu, x, n, out) != 0) return -1.0;
    }
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        if (mu_gpu_vision_quick_gelu_bf16(gpu, x, n, out) != 0) return -2.0;
    }
    return (now_ms() - t0) / (double)iters;
}

static double bench_add(mu_gpu *gpu, const float *a, const float *b,
                        float *out, int n, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) {
        if (mu_gpu_vision_add_bf16(gpu, a, b, n, out) != 0) return -1.0;
    }
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) {
        if (mu_gpu_vision_add_bf16(gpu, a, b, n, out) != 0) return -2.0;
    }
    return (now_ms() - t0) / (double)iters;
}

static int run_vision_ops(mu_gpu *gpu, int rows, int warmup, int iters) {
    size_t embed_count = (size_t)rows * 1280u;
    size_t kv_count = (size_t)rows * 2560u;
    size_t rotary_count = (size_t)rows * 40u;
    size_t mlp_count = (size_t)rows * 5120u;
    float *a = (float *)malloc(mlp_count * sizeof(float));
    float *b = (float *)malloc(mlp_count * sizeof(float));
    float *kv = (float *)malloc(kv_count * sizeof(float));
    float *rotary = (float *)malloc(rotary_count * sizeof(float));
    float *out = (float *)malloc(mlp_count * sizeof(float));
    uint16_t *weight = (uint16_t *)malloc(1280u * sizeof(uint16_t));
    uint16_t *bias = (uint16_t *)malloc(1280u * sizeof(uint16_t));
    if (!a || !b || !kv || !rotary || !out || !weight || !bias) {
        free(a); free(b); free(kv); free(rotary); free(out); free(weight); free(bias);
        return 1;
    }

    uint32_t state = 0x9e3779b9u ^ (uint32_t)rows;
    for (size_t i = 0; i < mlp_count; i++) {
        a[i] = round_bf16(random_small(&state));
        b[i] = round_bf16(random_small(&state));
        out[i] = 0.0f;
    }
    for (size_t i = 0; i < kv_count; i++) kv[i] = round_bf16(random_small(&state));
    for (size_t i = 0; i < rotary_count; i++) rotary[i] = random_small(&state);
    for (int i = 0; i < 1280; i++) {
        weight[i] = f32_to_bf16(1.0f + random_small(&state));
        bias[i] = f32_to_bf16(random_small(&state));
    }

    double vision_layernorm_ms =
        bench_layernorm(gpu, a, weight, bias, out, rows, warmup, iters);
    double vision_attn_ms =
        bench_attn(gpu, a, kv, rotary, out, rows, warmup, iters);
    double vision_gelu_ms =
        bench_gelu(gpu, a, out, (int)mlp_count, warmup, iters);
    double vision_add_ms =
        bench_add(gpu, a, b, out, (int)embed_count, warmup, iters);

    printf("vision_ops rows=%d vision_layernorm_ms=%.3f vision_attn_ms=%.3f "
           "vision_gelu_ms=%.3f vision_add_ms=%.3f checksum=%.6f\n",
           rows, vision_layernorm_ms, vision_attn_ms,
           vision_gelu_ms, vision_add_ms, checksum(out, embed_count));

    free(a); free(b); free(kv); free(rotary); free(out); free(weight); free(bias);
    return 0;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--rows N] [--iters N] [--warmup N] [--shape COLSxOUT_COLS] [--vision-ops]\n",
            argv0);
}

int main(int argc, char **argv) {
    int rows = 256;
    int iters = 3;
    int warmup = 1;
    int vision_ops = 0;
    bench_shape shapes[8] = {
        {1280, 1280},
        {1280, 2560},
        {1280, 5120},
        {5120, 1280},
    };
    int shape_count = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            rows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
            int c = 0, o = 0;
            if (sscanf(argv[++i], "%dx%d", &c, &o) != 2 || c <= 0 || o <= 0) {
                usage(argv[0]);
                return 2;
            }
            shapes[0].cols = c;
            shapes[0].out_cols = o;
            shape_count = 1;
        } else if (strcmp(argv[i], "--vision-ops") == 0) {
            vision_ops = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (rows <= 0 || iters <= 0 || warmup < 0) {
        usage(argv[0]);
        return 2;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "error: no default Metal device\n");
            return 1;
        }
        mu_gpu *gpu = NULL;
        if (mu_gpu_create(&gpu) != 0 || !mu_gpu_available(gpu)) {
            fprintf(stderr, "error: mu Metal backend unavailable\n");
            if (gpu) mu_gpu_destroy(gpu);
            return 1;
        }

        unsetenv("MU_DENSE_ROWS_SIMD");
        unsetenv("MU_DENSE_ROWS_TILED");

        printf("device=\"%s\" rows=%d iters=%d warmup=%d\n",
               mu_gpu_device_name(gpu), rows, iters, warmup);
        if (vision_ops) {
            int rc = run_vision_ops(gpu, rows, warmup, iters);
            mu_gpu_destroy(gpu);
            return rc;
        }

        for (int s = 0; s < shape_count; s++) {
            int cols = shapes[s].cols;
            int out_cols = shapes[s].out_cols;
            size_t x_count = (size_t)rows * (size_t)cols;
            size_t w_count = (size_t)out_cols * (size_t)cols;
            size_t out_count = (size_t)rows * (size_t)out_cols;

            float *x = (float *)malloc(x_count * sizeof(float));
            uint16_t *w_bf16 = (uint16_t *)malloc(w_count * sizeof(uint16_t));
            uint16_t *bias_bf16 = (uint16_t *)malloc((size_t)out_cols * sizeof(uint16_t));
            float *w_f32 = (float *)malloc(w_count * sizeof(float));
            float *cpu_out = (float *)malloc(out_count * sizeof(float));
            float *mps_out = (float *)malloc(out_count * sizeof(float));
            float *metal_out = (float *)malloc(out_count * sizeof(float));
            if (!x || !w_bf16 || !bias_bf16 || !w_f32 || !cpu_out || !mps_out || !metal_out) {
                fprintf(stderr, "error: allocation failed for %dx%d rows=%d\n",
                        cols, out_cols, rows);
                free(x);
                free(w_bf16);
                free(bias_bf16);
                free(w_f32);
                free(cpu_out);
                free(mps_out);
                free(metal_out);
                mu_gpu_destroy(gpu);
                return 1;
            }

            fill_inputs(x, w_bf16, bias_bf16, rows, cols, out_cols);
            bf16_to_f32_array(w_bf16, w_f32, w_count);

            double cpu_ms = bench_cpu_sgemm(x, w_f32, cpu_out,
                                            rows, cols, out_cols,
                                            warmup, iters);
            double mps_ms = bench_mps(device, x, w_f32, mps_out,
                                      rows, cols, out_cols,
                                      warmup, iters);
            double metal_current_ms = bench_metal_current(gpu, x, w_bf16, bias_bf16,
                                                          rows, cols, out_cols,
                                                          warmup, iters);
            int metal_rc = mu_gpu_dense_bf16_bias_rows(gpu, x, w_bf16, bias_bf16,
                                                       rows, cols, out_cols,
                                                       metal_out);
            double mps_diff = mps_ms >= 0.0 ? max_abs_diff(cpu_out, mps_out, out_count) : -1.0;
            double metal_diff = metal_rc == 0 ? max_abs_diff(cpu_out, metal_out, out_count) : -1.0;
            double gflop = 2.0 * (double)rows * (double)cols * (double)out_cols / 1.0e9;

            printf("shape=%dx%d cpu_sgemm_ms=%.3f mps_ms=%.3f metal_current_ms=%.3f "
                   "mps_vs_cpu=%.2fx metal_vs_cpu=%.2fx gflop=%.3f "
                   "mps_max_abs=%.6g metal_max_abs=%.6g checksum=%.6f\n",
                   cols, out_cols, cpu_ms, mps_ms, metal_current_ms,
                   mps_ms > 0.0 ? mps_ms / cpu_ms : -1.0,
                   metal_current_ms > 0.0 ? metal_current_ms / cpu_ms : -1.0,
                   gflop, mps_diff, metal_diff, checksum(cpu_out, out_count));

            free(x);
            free(w_bf16);
            free(bias_bf16);
            free(w_f32);
            free(cpu_out);
            free(mps_out);
            free(metal_out);
        }

        mu_gpu_destroy(gpu);
    }
    return 0;
}
