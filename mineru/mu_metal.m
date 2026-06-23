#include "mu_gpu.h"

struct mu_engine;
typedef struct mu_engine mu_engine;
const unsigned short *mu_engine_get_vision_block_tensor(void *engine, int layer, const char *suffix, int ndim, unsigned long d0, unsigned long d1);
const unsigned short *mu_engine_get_vision_merger_tensor(void *engine, const char *name, int ndim, unsigned long d0, unsigned long d1);
const unsigned short *mu_engine_get_text_layer_tensor(void *engine, int layer, const char *suffix, int ndim, unsigned long d0, unsigned long d1);
int mu_engine_vision_layers(const mu_engine *e);
int mu_engine_text_layers(const mu_engine *e);
int mu_engine_hidden_size(const mu_engine *e);

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <sys/time.h>

static double local_time_now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

@protocol MTLCommandBufferProfiling <NSObject>
@property (readonly) double kernelStartTime;
@property (readonly) double kernelEndTime;
@property (readonly) double GPUStartTime;
@property (readonly) double GPUEndTime;
@end

#define MU_GPU_WEIGHT_CACHE_CAP 1024
#define MU_GPU_DENSE_MPS_WEIGHT_CACHE_CAP 256

typedef struct {
    const void *cpu_ptr;
    NSUInteger length;
    id<MTLBuffer> buffer;
} mu_gpu_cached_buffer;

typedef struct {
    id<MTLBuffer> src;
    NSUInteger offset;
    NSUInteger length;
    id<MTLBuffer> f32;
} mu_gpu_dense_mps_weight;

struct mu_gpu {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> dense_probe;
    id<MTLComputePipelineState> dense_bf16_bias_probe;
    id<MTLComputePipelineState> dense_f32_bias_probe;
    id<MTLComputePipelineState> dense_f32_rows;
    id<MTLComputePipelineState> dense_f32_bias_rows;
    id<MTLComputePipelineState> dense_bf16_bias_rows;
    id<MTLComputePipelineState> dense_mps_bias_round;
    id<MTLComputePipelineState> rmsnorm_probe;
    id<MTLComputePipelineState> rmsnorm_bf16_probe;
    id<MTLComputePipelineState> rmsnorm_bf16_rows;
    id<MTLComputePipelineState> layernorm_bf16_probe;
    id<MTLComputePipelineState> layernorm_bf16_rows;
    id<MTLComputePipelineState> text_attn_token0;
    id<MTLComputePipelineState> text_attn_seq;
    id<MTLComputePipelineState> text_attn_seq_pos;
    id<MTLComputePipelineState> text_attn_cached;
    id<MTLComputePipelineState> text_rope_cache_update;
    id<MTLComputePipelineState> text_prefill_rope_cache_update;
    id<MTLComputePipelineState> add_f32;
    id<MTLComputePipelineState> silu_mul_f32;
    id<MTLComputePipelineState> vision_attn_concat_probe;
    id<MTLComputePipelineState> vision_qk_scores_head;
    id<MTLComputePipelineState> vision_rope_qk_rows;
    id<MTLComputePipelineState> vision_qk_scores_head_prerot;
    id<MTLComputePipelineState> vision_softmax_bf16_rows;
    id<MTLComputePipelineState> vision_pv_head;
    id<MTLComputePipelineState> vision_softmax_pv_head;
    id<MTLComputePipelineState> vision_attn_rows_online;
    id<MTLComputePipelineState> vision_attn_rows_flash;
    id<MTLComputePipelineState> vision_attn_rows_flash_k16;
    id<MTLComputePipelineState> vision_attn_pack_qkv_mpsgraph;
    id<MTLComputePipelineState> vision_attn_copy_mpsgraph;
    id<MTLComputePipelineState> vision_add_bf16;
    id<MTLComputePipelineState> vision_quick_gelu_bf16;
    id<MTLComputePipelineState> vision_gelu_bf16;
    id<MTLComputePipelineState> vision_merge4;
    id<MTLComputePipelineState> argmax_f32;
    id<MTLComputePipelineState> dense_probe_simd;
    id<MTLComputePipelineState> dense_probe_add_simd;
    id<MTLComputePipelineState> dense_bf16_bias_probe_simd;
    id<MTLComputePipelineState> dense_f32_bias_probe_simd;
    id<MTLComputePipelineState> dense_bf16_bias_rows_simd;
    id<MTLComputePipelineState> dense_bf16_bias_rows_tiled;
    id<MTLComputePipelineState> dense_bf16_bias_rows_simdgroup;
    id<MTLComputePipelineState> dense_bf16_bias_rows_simdgroup_quick_gelu;
    id<MTLComputePipelineState> dense_bf16_bias_rows_simdgroup_gelu;
    id<MTLComputePipelineState> dense_bf16_bias_rows_simdgroup_qkv;
    id<MTLComputePipelineState> text_decode_fused_ffn;
    id<MTLComputePipelineState> text_decode_qkv_proj_simd;
    id<MTLComputePipelineState> text_decode_qkv_rope_cache_simd;
    id<MTLComputePipelineState> text_attn_cached_simd;
    id<MTLComputePipelineState> text_prefill_attn_flash;
    id<MTLComputePipelineState> text_prefill_attn_pos_flash;
    MPSMatrixMultiplication *dense_mps_1280_1280;
    MPSMatrixMultiplication *dense_mps_1280_2560;
    MPSMatrixMultiplication *dense_mps_1280_3840;
    MPSMatrixMultiplication *dense_mps_1280_5120;
    MPSMatrixMultiplication *dense_mps_5120_1280;
    MPSMatrixMultiplication *dense_mps_896_896;
    MPSMatrixMultiplication *dense_mps_896_128;
    MPSMatrixMultiplication *dense_mps_896_4864;
    MPSMatrixMultiplication *dense_mps_4864_896;
    MPSGraph *vision_attn_mpsgraph;
    MPSGraphTensor *vision_attn_mpsgraph_q;
    MPSGraphTensor *vision_attn_mpsgraph_k;
    MPSGraphTensor *vision_attn_mpsgraph_v;
    MPSGraphTensor *vision_attn_mpsgraph_out;
    int vision_attn_mpsgraph_rows;
    int dense_mps_1280_1280_rows;
    int dense_mps_1280_2560_rows;
    int dense_mps_1280_3840_rows;
    int dense_mps_1280_5120_rows;
    int dense_mps_5120_1280_rows;
    int dense_mps_896_896_rows;
    int dense_mps_896_128_rows;
    int dense_mps_896_4864_rows;
    int dense_mps_4864_896_rows;
    char device_name[256];

    // Weight Buffer Cache
    mu_gpu_cached_buffer weight_cache[MU_GPU_WEIGHT_CACHE_CAP];
    int weight_cache_count;
    long long weight_cache_hits;
    long long weight_cache_misses;
    long long weight_cache_no_copy_allocs;
    long long weight_cache_copy_allocs;
    mu_gpu_dense_mps_weight dense_mps_weight_cache[MU_GPU_DENSE_MPS_WEIGHT_CACHE_CAP];
    int dense_mps_weight_cache_count;

    // Scratchpad Activation Arena
    id<MTLBuffer> scratch_a;
    id<MTLBuffer> scratch_b;
    NSUInteger scratch_size;
};

struct mu_gpu_kv_cache {
    mu_gpu *gpu;
    id<MTLBuffer> k_cache;
    id<MTLBuffer> v_cache;
    int layers;
    int cap;
};

static bool mu_gpu_dense_mps_shape(int cols, int out_cols);

static MPSMatrixMultiplication *mu_gpu_dense_mps_kernel(mu_gpu *gpu,
                                                       int rows, int cols, int out_cols);
static int mu_gpu_vision_attn_rows_mpsgraph_stage(mu_gpu *gpu,
                                                  mu_gpu_cmd_ctx **ctx,
                                                  mu_gpu_buf q,
                                                  mu_gpu_buf kv,
                                                  const float *rotary,
                                                  int rows,
                                                  mu_gpu_buf out,
                                                  bool shape_profile);

static NSString *mu_gpu_shader_path(NSString *name) {
    NSString *cwd_path = [@"mineru/metal" stringByAppendingPathComponent:name];
    if ([[NSFileManager defaultManager] fileExistsAtPath:cwd_path]) return cwd_path;

    NSString *src = [NSString stringWithUTF8String:__FILE__];
    NSString *dir = [src stringByDeletingLastPathComponent];
    NSString *from_src = [[dir stringByAppendingPathComponent:@"metal"]
        stringByAppendingPathComponent:name];
    if ([[NSFileManager defaultManager] fileExistsAtPath:from_src]) return from_src;

    return cwd_path;
}

static id<MTLComputePipelineState> mu_gpu_make_pipeline(id<MTLDevice> device,
                                                        NSString *source_name,
                                                        NSString *function_name) {
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:mu_gpu_shader_path(source_name)
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    if (!source) {
        if (getenv("MU_METAL_DEBUG")) {
            fprintf(stderr, "mu metal load failed: %s %s\n",
                    [source_name UTF8String],
                    error ? [[error localizedDescription] UTF8String] : "unknown");
        }
        return nil;
    }

    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        if (getenv("MU_METAL_DEBUG")) {
            fprintf(stderr, "mu metal compile failed: %s %s\n",
                    [source_name UTF8String],
                    error ? [[error localizedDescription] UTF8String] : "unknown");
        }
        return nil;
    }

    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (!function) {
        if (getenv("MU_METAL_DEBUG")) {
            fprintf(stderr, "mu metal function missing: %s %s\n",
                    [source_name UTF8String], [function_name UTF8String]);
        }
        return nil;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline && getenv("MU_METAL_DEBUG")) {
        fprintf(stderr, "mu metal pipeline failed: %s %s %s\n",
                [source_name UTF8String], [function_name UTF8String],
                error ? [[error localizedDescription] UTF8String] : "unknown");
    }
    return pipeline;
}

static float mu_gpu_bf16_to_f32(unsigned short v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static bool mu_gpu_dense_mps_text_shape(int cols, int out_cols);
static id<MTLBuffer> mu_gpu_dense_mps_f32_weight(mu_gpu *gpu, id<MTLBuffer> src,
                                                NSUInteger offset, NSUInteger length);
static int mu_gpu_dense_f32_rows_mps(mu_gpu *gpu,
                                     id<MTLBuffer> x_buf, NSUInteger x_offset,
                                     id<MTLBuffer> w_buf, NSUInteger w_offset,
                                     id<MTLBuffer> out_buf, NSUInteger out_offset,
                                     int x_rows, int cols, int out_cols,
                                     float *out);

static id<MTLBuffer> mu_gpu_get_or_create_buffer(mu_gpu *gpu, const void *cpu_ptr, NSUInteger length) {
    if (!gpu || !cpu_ptr || length == 0) return nil;

    // Check if it's already cached
    for (int i = 0; i < gpu->weight_cache_count; i++) {
        if (gpu->weight_cache[i].cpu_ptr == cpu_ptr) {
            gpu->weight_cache_hits++;
            return gpu->weight_cache[i].buffer;
        }
    }

    gpu->weight_cache_misses++;

    if (gpu->weight_cache_count >= MU_GPU_WEIGHT_CACHE_CAP) {
        if (getenv("MU_METAL_DEBUG")) {
            fprintf(stderr, "Warning: Metal weight cache capacity reached (%d)\n", MU_GPU_WEIGHT_CACHE_CAP);
        }
        gpu->weight_cache_copy_allocs++;
        return [gpu->device newBufferWithBytes:cpu_ptr length:length options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> buffer = nil;
    static int page_size = 0;
    if (page_size == 0) {
        page_size = getpagesize();
        if (page_size <= 0) page_size = 16384;
    }

    if (((uintptr_t)cpu_ptr) % page_size == 0) {
        buffer = [gpu->device newBufferWithBytesNoCopy:(void *)cpu_ptr
                                                length:length
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
        if (buffer) {
            gpu->weight_cache_no_copy_allocs++;
        }
    }

    if (!buffer) {
        buffer = [gpu->device newBufferWithBytes:cpu_ptr
                                          length:length
                                         options:MTLResourceStorageModeShared];
        if (buffer) {
            gpu->weight_cache_copy_allocs++;
        }
    }

    if (buffer) {
        gpu->weight_cache[gpu->weight_cache_count].cpu_ptr = cpu_ptr;
        gpu->weight_cache[gpu->weight_cache_count].length = length;
        gpu->weight_cache[gpu->weight_cache_count].buffer = buffer;
        gpu->weight_cache_count++;
    }

    if (!buffer) {
        if (getenv("MU_METAL_DEBUG")) {
            fprintf(stderr, "Error: mu_gpu_get_or_create_buffer failed to allocate buffer of length %lu, cpu_ptr = %p\n", (unsigned long)length, cpu_ptr);
        }
    }

    return buffer;
}

typedef struct {
    mu_gpu *gpu;
    NSUInteger offset_a;
    NSUInteger offset_b;
} mu_scratch_allocator;

static id<MTLBuffer> mu_scratch_alloc_a(mu_scratch_allocator *alloc, NSUInteger size, NSUInteger *out_offset) {
    NSUInteger aligned = (size + 255) & ~255;
    if (alloc->offset_a + aligned <= alloc->gpu->scratch_size) {
        *out_offset = alloc->offset_a;
        alloc->offset_a += aligned;
        return alloc->gpu->scratch_a;
    }
    if (getenv("MU_METAL_DEBUG")) {
        fprintf(stderr, "Error: mu_scratch_alloc_a failed: offset_a=%lu, aligned=%lu, scratch_size=%lu\n",
                (unsigned long)alloc->offset_a, (unsigned long)aligned, (unsigned long)alloc->gpu->scratch_size);
    }
    *out_offset = 0;
    return nil;
}

static id<MTLBuffer> mu_scratch_alloc_b(mu_scratch_allocator *alloc, NSUInteger size, NSUInteger *out_offset) {
    NSUInteger aligned = (size + 255) & ~255;
    if (alloc->offset_b + aligned <= alloc->gpu->scratch_size) {
        *out_offset = alloc->offset_b;
        alloc->offset_b += aligned;
        return alloc->gpu->scratch_b;
    }
    *out_offset = 0;
    return nil;
}

struct mu_gpu_cmd_ctx {
    mu_gpu *gpu;
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    mu_scratch_allocator alloc;
};

int mu_gpu_cmd_begin_with_scratch_offsets(mu_gpu *gpu, unsigned long offset_a,
                                          unsigned long offset_b,
                                          mu_gpu_cmd_ctx **out_ctx) {
    if (!gpu || !out_ctx) return -1;
    if (offset_a > gpu->scratch_size || offset_b > gpu->scratch_size) return -5;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
        if (!cb) return -2;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return -3;

        mu_gpu_cmd_ctx *ctx = (mu_gpu_cmd_ctx *)malloc(sizeof(*ctx));
        if (!ctx) return -4;
        ctx->gpu = gpu;
        ctx->command_buffer = cb;
        ctx->encoder = enc;
        ctx->alloc.gpu = gpu;
        ctx->alloc.offset_a = (NSUInteger)offset_a;
        ctx->alloc.offset_b = (NSUInteger)offset_b;
        *out_ctx = ctx;
    }
    return 0;
}

int mu_gpu_cmd_begin(mu_gpu *gpu, mu_gpu_cmd_ctx **out_ctx) {
    return mu_gpu_cmd_begin_with_scratch_offsets(gpu, 0, 0, out_ctx);
}

void mu_gpu_cmd_set_label(mu_gpu_cmd_ctx *ctx, const char *label) {
    if (ctx && ctx->command_buffer && label) {
        @autoreleasepool {
            ctx->command_buffer.label = [NSString stringWithUTF8String:label];
        }
    }
}

void mu_gpu_cmd_get_scratch_offsets(mu_gpu_cmd_ctx *ctx, unsigned long *offset_a,
                                    unsigned long *offset_b) {
    if (offset_a) *offset_a = ctx ? (unsigned long)ctx->alloc.offset_a : 0;
    if (offset_b) *offset_b = ctx ? (unsigned long)ctx->alloc.offset_b : 0;
}

int mu_gpu_cmd_commit_and_wait(mu_gpu_cmd_ctx *ctx) {
    if (!ctx) return -1;
    int rc = 0;
    @autoreleasepool {
        if (ctx->encoder) [ctx->encoder endEncoding];

        static int check_profile = -1;
        if (check_profile == -1) {
            const char *env = getenv("MU_LATENCY_PROFILE");
            check_profile = (env && strcmp(env, "0") != 0) ? 1 : 0;
        }

        double t_commit_start = 0.0;
        double t_commit_end = 0.0;
        double t_wait_end = 0.0;

        if (check_profile) {
            t_commit_start = local_time_now_seconds();
        }

        [ctx->command_buffer commit];

        if (check_profile) {
            t_commit_end = local_time_now_seconds();
        }

        [ctx->command_buffer waitUntilCompleted];

        if (check_profile) {
            t_wait_end = local_time_now_seconds();
            double kernel_start = 0.0;
            double kernel_end = 0.0;
            double gpu_start = 0.0;
            double gpu_end = 0.0;

            id<MTLCommandBufferProfiling> cb = (id<MTLCommandBufferProfiling>)ctx->command_buffer;
            if ([cb respondsToSelector:@selector(kernelStartTime)]) {
                kernel_start = cb.kernelStartTime;
            }
            if ([cb respondsToSelector:@selector(kernelEndTime)]) {
                kernel_end = cb.kernelEndTime;
            }
            if ([cb respondsToSelector:@selector(GPUStartTime)]) {
                gpu_start = cb.GPUStartTime;
            }
            if ([cb respondsToSelector:@selector(GPUEndTime)]) {
                gpu_end = cb.GPUEndTime;
            }

            NSString *label = ctx->command_buffer.label;
            const char *label_str = label ? [label UTF8String] : "unlabeled";

            if (gpu_start > 0.0 && gpu_end > 0.0) {
                fprintf(stderr, "[MU_LATENCY_PROFILE] command_buffer='%s'\n"
                                "  CPU Enqueue (Commit call): %10.3f ms\n"
                                "  Driver Scheduling Delay:  %10.3f ms\n"
                                "  Queue Handoff Latency:    %10.3f ms\n"
                                "  GPU Execution Time:       %10.3f ms\n"
                                "  CPU Wait/Stall Time:      %10.3f ms\n"
                                "  Total Command Buffer Lft: %10.3f ms\n",
                        label_str,
                        (t_commit_end - t_commit_start) * 1000.0,
                        (kernel_end - kernel_start) * 1000.0,
                        (gpu_start - kernel_end) * 1000.0,
                        (gpu_end - gpu_start) * 1000.0,
                        (t_wait_end - t_commit_end) * 1000.0,
                        (t_wait_end - t_commit_start) * 1000.0);
            } else {
                fprintf(stderr, "[MU_LATENCY_PROFILE] command_buffer='%s'\n"
                                "  CPU Enqueue (Commit call): %10.3f ms\n"
                                "  Driver Scheduling Delay:  %10.3f ms\n"
                                "  Queue Handoff Latency:           n/a (GPU timestamp unavailable)\n"
                                "  GPU Execution Time:              n/a (GPU timestamp unavailable)\n"
                                "  CPU Wait/Stall Time:      %10.3f ms\n"
                                "  Total Command Buffer Lft: %10.3f ms\n",
                        label_str,
                        (t_commit_end - t_commit_start) * 1000.0,
                        (kernel_end - kernel_start) * 1000.0,
                        (t_wait_end - t_commit_end) * 1000.0,
                        (t_wait_end - t_commit_start) * 1000.0);
            }
        }

        if (ctx->command_buffer.status != MTLCommandBufferStatusCompleted) {
            rc = -2;
        }
        ctx->encoder = nil;
        ctx->command_buffer = nil;
        free(ctx);
    }
    return rc;
}

void mu_gpu_cmd_discard(mu_gpu_cmd_ctx *ctx) {
    if (!ctx) return;
    @autoreleasepool {
        if (ctx->encoder) [ctx->encoder endEncoding];
        ctx->encoder = nil;
        ctx->command_buffer = nil;
        free(ctx);
    }
}

static int mu_gpu_profile_commit_stage(mu_gpu *gpu, mu_gpu_cmd_ctx **ctx,
                                       const char *stage,
                                       const char *next_label) {
    if (!gpu || !ctx || !*ctx || !stage) return -1;
    unsigned long offset_a = 0;
    unsigned long offset_b = 0;
    mu_gpu_cmd_get_scratch_offsets(*ctx, &offset_a, &offset_b);

    double start = local_time_now_seconds();
    int rc = mu_gpu_cmd_commit_and_wait(*ctx);
    double seconds = local_time_now_seconds() - start;
    *ctx = NULL;

    fprintf(stderr, "mu_timing stage=%s seconds=%.6f\n", stage, seconds);

    if (rc != 0 || !next_label) return rc;

    rc = mu_gpu_cmd_begin_with_scratch_offsets(gpu, offset_a, offset_b, ctx);
    if (rc != 0) return rc;
    mu_gpu_cmd_set_label(*ctx, next_label);
    return 0;
}

static int mu_gpu_cmd_end_encoder(mu_gpu_cmd_ctx *ctx) {
    if (!ctx) return -1;
    if (ctx->encoder) {
        [ctx->encoder endEncoding];
        ctx->encoder = nil;
    }
    return 0;
}

static int mu_gpu_cmd_begin_encoder(mu_gpu_cmd_ctx *ctx) {
    if (!ctx || !ctx->command_buffer) return -1;
    if (ctx->encoder) return 0;
    ctx->encoder = [ctx->command_buffer computeCommandEncoder];
    return ctx->encoder ? 0 : -2;
}

mu_gpu_buf mu_gpu_get_weight_buf(mu_gpu *gpu, const void *cpu_ptr, unsigned long length) {
    mu_gpu_buf res = { NULL, 0 };
    if (!gpu) return res;
    id<MTLBuffer> buf = mu_gpu_get_or_create_buffer(gpu, cpu_ptr, length);
    res.ptr = (__bridge void *)buf;
    res.offset = 0;
    return res;
}

mu_gpu_buf mu_gpu_scratch_b_at(mu_gpu *gpu, unsigned long offset, unsigned long size) {
    mu_gpu_buf res = { NULL, 0 };
    if (!gpu || !gpu->scratch_b || offset + size > gpu->scratch_size) return res;
    res.ptr = (__bridge void *)gpu->scratch_b;
    res.offset = offset;
    return res;
}

mu_gpu_buf mu_gpu_scratch_alloc_a_ctx(mu_gpu_cmd_ctx *ctx, unsigned long size) {
    mu_gpu_buf res = { NULL, 0 };
    if (!ctx) return res;
    NSUInteger offset = 0;
    id<MTLBuffer> buf = mu_scratch_alloc_a(&ctx->alloc, size, &offset);
    if (buf) {
        res.ptr = (__bridge void *)buf;
        res.offset = offset;
    }
    return res;
}

mu_gpu_buf mu_gpu_scratch_alloc_b_ctx(mu_gpu_cmd_ctx *ctx, unsigned long size) {
    mu_gpu_buf res = { NULL, 0 };
    if (!ctx) return res;
    NSUInteger offset = 0;
    id<MTLBuffer> buf = mu_scratch_alloc_b(&ctx->alloc, size, &offset);
    if (buf) {
        res.ptr = (__bridge void *)buf;
        res.offset = offset;
    }
    return res;
}

void mu_gpu_buf_copy_to(mu_gpu_buf dst, const void *src, unsigned long size) {
    if (!dst.ptr || !src || size == 0) return;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)dst.ptr;
    memcpy((char *)[buf contents] + dst.offset, src, size);
}

void mu_gpu_buf_copy_from(void *dst, mu_gpu_buf src, unsigned long size) {
    if (!dst || !src.ptr || size == 0) return;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)src.ptr;
    memcpy(dst, (char *)[buf contents] + src.offset, size);
}

static int mu_gpu_vision_attn_mpsgraph_ensure(mu_gpu *gpu, int rows) {
    if (!gpu || rows <= 0) return -1;
    if (gpu->vision_attn_mpsgraph && gpu->vision_attn_mpsgraph_rows == rows) return 0;

    MPSGraph *graph = [MPSGraph new];
    graph.options = MPSGraphOptionsNone;
    MPSShape *packed_shape = @[ @1, @16, @(rows), @80 ];
    MPSShape *out_shape = @[ @(rows), @1280 ];

    MPSGraphTensor *q = [graph placeholderWithShape:packed_shape dataType:MPSDataTypeFloat32 name:@"q"];
    MPSGraphTensor *k = [graph placeholderWithShape:packed_shape dataType:MPSDataTypeFloat32 name:@"k"];
    MPSGraphTensor *v = [graph placeholderWithShape:packed_shape dataType:MPSDataTypeFloat32 name:@"v"];
    MPSGraphTensor *attn = [graph scaledDotProductAttentionWithQueryTensor:q
                                                                  keyTensor:k
                                                                valueTensor:v
                                                                      scale:0.11180339887498949f
                                                                       name:@"sdpa"];
    MPSGraphTensor *transposed = [graph transposeTensor:attn permutation:@[ @0, @2, @1, @3 ] name:@"sdpa_nh"];
    MPSGraphTensor *out = [graph reshapeTensor:transposed withShape:out_shape name:@"out"];
    if (!q || !k || !v || !out) return -2;

    gpu->vision_attn_mpsgraph = graph;
    gpu->vision_attn_mpsgraph_q = q;
    gpu->vision_attn_mpsgraph_k = k;
    gpu->vision_attn_mpsgraph_v = v;
    gpu->vision_attn_mpsgraph_out = out;
    gpu->vision_attn_mpsgraph_rows = rows;
    return 0;
}

static int mu_gpu_vision_attn_rows_mpsgraph_stage(mu_gpu *gpu,
                                                  mu_gpu_cmd_ctx **ctx,
                                                  mu_gpu_buf q,
                                                  mu_gpu_buf kv,
                                                  const float *rotary,
                                                  int rows,
                                                  mu_gpu_buf out,
                                                  bool shape_profile) {
    if (!gpu || !ctx || !*ctx || !q.ptr || !kv.ptr || !rotary || !out.ptr || rows <= 0) return -1;
    if (!gpu->vision_attn_pack_qkv_mpsgraph || !gpu->vision_attn_copy_mpsgraph) return -2;

    bool mpsgraph_profile = getenv("MU_VISION_ATTN_MPSGRAPH_PROFILE") != NULL;
    unsigned long offset_a = 0;
    unsigned long offset_b = 0;
    if (mpsgraph_profile) {
        mu_gpu_cmd_get_scratch_offsets(*ctx, &offset_a, &offset_b);
        double t_boundary = local_time_now_seconds();
        int rc = mu_gpu_cmd_commit_and_wait(*ctx);
        fprintf(stderr, "mu_timing stage=vision_attn_mpsgraph_prepack_boundary seconds=%.6f\n",
                local_time_now_seconds() - t_boundary);
        *ctx = NULL;
        if (rc != 0) return rc;
        rc = mu_gpu_cmd_begin_with_scratch_offsets(gpu, offset_a, offset_b, ctx);
        if (rc != 0) return rc;
        mu_gpu_cmd_set_label(*ctx, "vision_attn_mpsgraph_pack_qkv");
    }

    id<MTLBuffer> q_buf = (__bridge id<MTLBuffer>)q.ptr;
    id<MTLBuffer> kv_buf = (__bridge id<MTLBuffer>)kv.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;
    NSUInteger packed_bytes = (NSUInteger)rows * 1280u * sizeof(float);
    NSUInteger rotary_bytes = (NSUInteger)rows * 40u * sizeof(float);
    double t_alloc = local_time_now_seconds();
    id<MTLBuffer> rotary_buf = [gpu->device newBufferWithBytes:rotary
                                                        length:rotary_bytes
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> q_pack = [gpu->device newBufferWithLength:packed_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> k_pack = [gpu->device newBufferWithLength:packed_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> v_pack = [gpu->device newBufferWithLength:packed_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> graph_out = [gpu->device newBufferWithLength:packed_bytes options:MTLResourceStorageModeShared];
    if (mpsgraph_profile) {
        fprintf(stderr, "mu_timing stage=vision_attn_mpsgraph_buffer_alloc seconds=%.6f\n",
                local_time_now_seconds() - t_alloc);
    }
    if (!rotary_buf || !q_pack || !k_pack || !v_pack || !graph_out) return -3;

    [(*ctx)->encoder setComputePipelineState:gpu->vision_attn_pack_qkv_mpsgraph];
    [(*ctx)->encoder setBuffer:q_buf offset:q.offset atIndex:0];
    [(*ctx)->encoder setBuffer:kv_buf offset:kv.offset atIndex:1];
    [(*ctx)->encoder setBuffer:rotary_buf offset:0 atIndex:2];
    [(*ctx)->encoder setBuffer:q_pack offset:0 atIndex:3];
    [(*ctx)->encoder setBuffer:k_pack offset:0 atIndex:4];
    [(*ctx)->encoder setBuffer:v_pack offset:0 atIndex:5];
    [(*ctx)->encoder setBytes:&rows length:sizeof(rows) atIndex:6];
    MTLSize pack_grid = MTLSizeMake((NSUInteger)rows * 1280u, 1, 1);
    MTLSize pack_threads = MTLSizeMake(256, 1, 1);
    [(*ctx)->encoder dispatchThreads:pack_grid threadsPerThreadgroup:pack_threads];

    mu_gpu_cmd_get_scratch_offsets(*ctx, &offset_a, &offset_b);
    double t_pack = local_time_now_seconds();
    int rc = mu_gpu_cmd_commit_and_wait(*ctx);
    if (mpsgraph_profile) {
        fprintf(stderr, "mu_timing stage=vision_attn_mpsgraph_pack_qkv seconds=%.6f\n",
                local_time_now_seconds() - t_pack);
    }
    *ctx = NULL;
    if (rc != 0) return rc;

    double t_graph = local_time_now_seconds();
    rc = mu_gpu_vision_attn_mpsgraph_ensure(gpu, rows);
    if (rc != 0) return rc;

    MPSShape *packed_shape = @[ @1, @16, @(rows), @80 ];
    MPSShape *out_shape = @[ @(rows), @1280 ];
    MPSGraphTensorData *q_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:q_pack
                                                                         shape:packed_shape
                                                                      dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *k_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:k_pack
                                                                         shape:packed_shape
                                                                      dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *v_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:v_pack
                                                                         shape:packed_shape
                                                                      dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *out_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:graph_out
                                                                           shape:out_shape
                                                                        dataType:MPSDataTypeFloat32];
    if (!q_data || !k_data || !v_data || !out_data) return -4;

    @try {
        MPSGraphTensorDataDictionary *feeds = @{
            gpu->vision_attn_mpsgraph_q: q_data,
            gpu->vision_attn_mpsgraph_k: k_data,
            gpu->vision_attn_mpsgraph_v: v_data,
        };
        MPSGraphTensorDataDictionary *results = @{ gpu->vision_attn_mpsgraph_out: out_data };
        [gpu->vision_attn_mpsgraph runWithMTLCommandQueue:gpu->queue
                                                     feeds:feeds
                                          targetOperations:nil
                                         resultsDictionary:results];
    } @catch (NSException *exception) {
        if (getenv("MU_METAL_DEBUG")) {
            fprintf(stderr, "mu mpsgraph attention failed: %s\n",
                    [[exception reason] UTF8String]);
        }
        return -5;
    }
    if (mpsgraph_profile) {
        fprintf(stderr, "mu_timing stage=vision_attn_mpsgraph_graph seconds=%.6f\n",
                local_time_now_seconds() - t_graph);
    }

    rc = mu_gpu_cmd_begin_with_scratch_offsets(gpu, offset_a, offset_b, ctx);
    if (rc != 0) return rc;
    if (mpsgraph_profile) mu_gpu_cmd_set_label(*ctx, "vision_attn_mpsgraph_copy_round");
    [(*ctx)->encoder setComputePipelineState:gpu->vision_attn_copy_mpsgraph];
    [(*ctx)->encoder setBuffer:graph_out offset:0 atIndex:0];
    [(*ctx)->encoder setBuffer:out_buf offset:out.offset atIndex:1];
    int n = rows * 1280;
    [(*ctx)->encoder setBytes:&n length:sizeof(n) atIndex:2];
    [(*ctx)->encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (mpsgraph_profile) {
        double t_copy = local_time_now_seconds();
        rc = mu_gpu_cmd_commit_and_wait(*ctx);
        fprintf(stderr, "mu_timing stage=vision_attn_mpsgraph_copy_round seconds=%.6f\n",
                local_time_now_seconds() - t_copy);
        *ctx = NULL;
        if (rc != 0) return rc;
        rc = mu_gpu_cmd_begin_with_scratch_offsets(gpu, offset_a, offset_b, ctx);
        if (rc != 0) return rc;
    }

    if (shape_profile) {
        fprintf(stderr,
                "mu_profile stage=vision_attn_shape path=mpsgraph rows=%d "
                "graph_shape=1x16x%dx80 heads=16\n",
                rows, rows);
    }
    return 0;
}

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
        gpu->dense_probe = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                @"mu_dense_probe");
        gpu->dense_bf16_bias_probe = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                          @"mu_dense_bf16_bias_probe");
        gpu->dense_f32_bias_probe = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                         @"mu_dense_f32_bias_probe");
        gpu->dense_f32_rows = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                   @"mu_dense_f32_rows");
        gpu->dense_f32_bias_rows = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                        @"mu_dense_f32_bias_rows");
        gpu->dense_bf16_bias_rows = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                         @"mu_dense_bf16_bias_rows");
        gpu->dense_mps_bias_round = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                         @"mu_dense_mps_bias_round");
        gpu->rmsnorm_probe = mu_gpu_make_pipeline(device, @"mu_norm.metal",
                                                  @"mu_rmsnorm_probe");
        gpu->rmsnorm_bf16_probe = mu_gpu_make_pipeline(device, @"mu_norm.metal",
                                                       @"mu_rmsnorm_bf16_probe");
        gpu->rmsnorm_bf16_rows = mu_gpu_make_pipeline(device, @"mu_norm.metal",
                                                      @"mu_rmsnorm_bf16_rows");
        gpu->layernorm_bf16_probe = mu_gpu_make_pipeline(device, @"mu_norm.metal",
                                                         @"mu_layernorm_bf16_probe");
        gpu->layernorm_bf16_rows = mu_gpu_make_pipeline(device, @"mu_norm.metal",
                                                        @"mu_layernorm_bf16_rows");
        gpu->text_attn_token0 = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                     @"mu_text_attn_token0");
        gpu->text_attn_seq = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                  @"mu_text_attn_seq");
        gpu->text_attn_seq_pos = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                      @"mu_text_attn_seq_pos");
        gpu->text_attn_cached = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                     @"mu_text_attn_cached");
        gpu->text_rope_cache_update = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                           @"mu_text_rope_cache_update");
        gpu->add_f32 = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                            @"mu_add_f32");
        gpu->silu_mul_f32 = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                 @"mu_silu_mul_f32");
        gpu->vision_attn_concat_probe = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                             @"mu_vision_attn_concat_probe");
        gpu->vision_qk_scores_head = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                          @"mu_vision_qk_scores_head");
        gpu->vision_rope_qk_rows = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                        @"mu_vision_rope_qk_rows");
        gpu->vision_qk_scores_head_prerot = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                                 @"mu_vision_qk_scores_head_prerot");
        gpu->vision_softmax_bf16_rows = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                             @"mu_vision_softmax_bf16_rows");
        gpu->vision_pv_head = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                   @"mu_vision_pv_head");
        gpu->vision_softmax_pv_head = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                           @"mu_vision_softmax_pv_head");
        gpu->vision_attn_rows_online = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                            @"mu_vision_attn_rows_online");
        gpu->vision_attn_rows_flash = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                           @"mu_vision_attn_rows_flash");
        if (getenv("MU_VISION_ATTN_FLASH_K16") != NULL) {
            gpu->vision_attn_rows_flash_k16 = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                                   @"mu_vision_attn_rows_flash_k16");
        }
        if (getenv("MU_VISION_ATTN_NO_MPSGRAPH") == NULL) {
            gpu->vision_attn_pack_qkv_mpsgraph = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                                      @"mu_vision_attn_pack_qkv_mpsgraph");
            gpu->vision_attn_copy_mpsgraph = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                                  @"mu_vision_attn_copy_mpsgraph");
        }
        gpu->vision_add_bf16 = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                    @"mu_vision_add_bf16");
        gpu->vision_quick_gelu_bf16 = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                           @"mu_vision_quick_gelu_bf16");
        gpu->vision_gelu_bf16 = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                     @"mu_vision_gelu_bf16");
        gpu->vision_merge4 = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                   @"mu_vision_merge4");
        gpu->argmax_f32 = mu_gpu_make_pipeline(device, @"mu_sample.metal",
                                               @"mu_argmax_f32");
        gpu->dense_probe_simd = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                     @"mu_dense_probe_simd");
        gpu->dense_probe_add_simd = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                         @"mu_dense_probe_add_simd");
        gpu->dense_bf16_bias_probe_simd = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                               @"mu_dense_bf16_bias_probe_simd");
        gpu->dense_f32_bias_probe_simd = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                              @"mu_dense_f32_bias_probe_simd");
        gpu->dense_bf16_bias_rows_simd = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                              @"mu_dense_bf16_bias_rows_simd");
        gpu->dense_bf16_bias_rows_tiled = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                               @"mu_dense_bf16_bias_rows_tiled");
        gpu->dense_bf16_bias_rows_simdgroup = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                                   @"mu_dense_bf16_bias_rows_simdgroup");
        gpu->dense_bf16_bias_rows_simdgroup_quick_gelu = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                                              @"mu_dense_bf16_bias_rows_simdgroup_quick_gelu");
        gpu->dense_bf16_bias_rows_simdgroup_gelu = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                                        @"mu_dense_bf16_bias_rows_simdgroup_gelu");
        gpu->dense_bf16_bias_rows_simdgroup_qkv = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                                       @"mu_dense_bf16_bias_rows_simdgroup_qkv");
        gpu->text_decode_fused_ffn = mu_gpu_make_pipeline(device, @"mu_text_fused_ffn.metal",
                                                          @"mu_text_decode_fused_ffn");
        gpu->text_decode_qkv_proj_simd = mu_gpu_make_pipeline(device, @"mu_dense.metal",
                                                              @"mu_text_decode_qkv_proj_simd");
        gpu->text_decode_qkv_rope_cache_simd = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                                    @"mu_text_decode_qkv_rope_cache_simd");
        gpu->text_attn_cached_simd = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                          @"mu_text_attn_cached_simd");
        gpu->text_prefill_rope_cache_update = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                                   @"mu_text_prefill_rope_cache_update");
        gpu->text_prefill_attn_flash = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                            @"mu_text_prefill_attn_flash");
        gpu->text_prefill_attn_pos_flash = mu_gpu_make_pipeline(device, @"mu_attn.metal",
                                                                @"mu_text_prefill_attn_pos_flash");
        const char *name = [[device name] UTF8String];
        if (name) {
            strlcpy(gpu->device_name, name, sizeof(gpu->device_name));
        } else {
            strlcpy(gpu->device_name, "unknown", sizeof(gpu->device_name));
        }
        gpu->scratch_size = 512 * 1024 * 1024; // 512 MB
        gpu->scratch_a = [device newBufferWithLength:gpu->scratch_size options:MTLResourceStorageModeShared];
        gpu->scratch_b = [device newBufferWithLength:gpu->scratch_size options:MTLResourceStorageModeShared];
        *out = gpu;
    }
    return 0;
}

void mu_gpu_destroy(mu_gpu *gpu) {
    if (!gpu) return;

    if (getenv("MU_METAL_DEBUG")) {
        fprintf(stderr, "=== Metal Weight Buffer Cache Stats ===\n");
        fprintf(stderr, "  Hits: %lld\n", gpu->weight_cache_hits);
        fprintf(stderr, "  Misses: %lld\n", gpu->weight_cache_misses);
        fprintf(stderr, "  Allocations (Zero-Copy): %lld\n", gpu->weight_cache_no_copy_allocs);
        fprintf(stderr, "  Allocations (Copy): %lld\n", gpu->weight_cache_copy_allocs);
        fprintf(stderr, "  Total Cached Buffers: %d\n", gpu->weight_cache_count);
        fprintf(stderr, "========================================\n");
    }

    gpu->scratch_a = nil;
    gpu->scratch_b = nil;
    for (int i = 0; i < gpu->weight_cache_count; i++) {
        gpu->weight_cache[i].buffer = nil;
    }
    for (int i = 0; i < gpu->dense_mps_weight_cache_count; i++) {
        gpu->dense_mps_weight_cache[i].src = nil;
        gpu->dense_mps_weight_cache[i].f32 = nil;
    }
    gpu->vision_merge4 = nil;
    gpu->argmax_f32 = nil;
    gpu->dense_probe_simd = nil;
    gpu->dense_probe_add_simd = nil;
    gpu->dense_bf16_bias_probe_simd = nil;
    gpu->dense_f32_bias_probe_simd = nil;
    gpu->dense_bf16_bias_rows_simd = nil;
    gpu->dense_bf16_bias_rows_tiled = nil;
    gpu->dense_bf16_bias_rows_simdgroup = nil;
    gpu->dense_bf16_bias_rows_simdgroup_quick_gelu = nil;
    gpu->dense_bf16_bias_rows_simdgroup_gelu = nil;
    gpu->dense_bf16_bias_rows_simdgroup_qkv = nil;
    gpu->text_decode_fused_ffn = nil;
    gpu->text_decode_qkv_proj_simd = nil;
    gpu->text_decode_qkv_rope_cache_simd = nil;
    gpu->text_attn_cached_simd = nil;
    gpu->dense_mps_1280_1280 = nil;
    gpu->dense_mps_1280_2560 = nil;
    gpu->dense_mps_1280_3840 = nil;
    gpu->dense_mps_1280_5120 = nil;
    gpu->dense_mps_5120_1280 = nil;
    gpu->dense_mps_896_896 = nil;
    gpu->dense_mps_896_128 = nil;
    gpu->dense_mps_896_4864 = nil;
    gpu->dense_mps_4864_896 = nil;
    gpu->vision_gelu_bf16 = nil;
    gpu->vision_quick_gelu_bf16 = nil;
    gpu->vision_add_bf16 = nil;
    gpu->vision_attn_rows_online = nil;
    gpu->vision_attn_rows_flash = nil;
    gpu->vision_attn_rows_flash_k16 = nil;
    gpu->vision_attn_pack_qkv_mpsgraph = nil;
    gpu->vision_attn_copy_mpsgraph = nil;
    gpu->vision_attn_mpsgraph = nil;
    gpu->vision_attn_mpsgraph_q = nil;
    gpu->vision_attn_mpsgraph_k = nil;
    gpu->vision_attn_mpsgraph_v = nil;
    gpu->vision_attn_mpsgraph_out = nil;
    gpu->vision_softmax_pv_head = nil;
    gpu->vision_pv_head = nil;
    gpu->vision_softmax_bf16_rows = nil;
    gpu->vision_qk_scores_head_prerot = nil;
    gpu->vision_rope_qk_rows = nil;
    gpu->vision_qk_scores_head = nil;
    gpu->vision_attn_concat_probe = nil;
    gpu->silu_mul_f32 = nil;
    gpu->add_f32 = nil;
    gpu->text_attn_seq_pos = nil;
    gpu->text_attn_cached = nil;
    gpu->text_rope_cache_update = nil;
    gpu->text_prefill_rope_cache_update = nil;
    gpu->text_prefill_attn_flash = nil;
    gpu->text_prefill_attn_pos_flash = nil;
    gpu->text_attn_seq = nil;
    gpu->text_attn_token0 = nil;
    gpu->layernorm_bf16_rows = nil;
    gpu->layernorm_bf16_probe = nil;
    gpu->rmsnorm_bf16_rows = nil;
    gpu->rmsnorm_bf16_probe = nil;
    gpu->rmsnorm_probe = nil;
    gpu->dense_mps_bias_round = nil;
    gpu->dense_bf16_bias_rows = nil;
    gpu->dense_f32_bias_rows = nil;
    gpu->dense_f32_rows = nil;
    gpu->dense_f32_bias_probe = nil;
    gpu->dense_bf16_bias_probe = nil;
    gpu->dense_probe = nil;
    gpu->queue = nil;
    gpu->device = nil;
    free(gpu);
}

bool mu_gpu_available(const mu_gpu *gpu) {
    return gpu && gpu->device;
}

const char *mu_gpu_device_name(const mu_gpu *gpu) {
    if (!gpu || !gpu->device_name[0]) return "none";
    return gpu->device_name;
}

int mu_gpu_dense_probe(mu_gpu *gpu, const float *x,
                       const unsigned short *w_bf16,
                       int rows, int cols, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger x_bytes = (NSUInteger)cols * sizeof(float);
    NSUInteger w_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(unsigned short);
    NSUInteger out_bytes = (NSUInteger)rows * sizeof(float);
    mu_gpu_buf x_buf = mu_gpu_scratch_alloc_a_ctx(ctx, x_bytes);
    mu_gpu_buf w_buf = mu_gpu_get_weight_buf(gpu, w_bf16, w_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!x_buf.ptr || !w_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(x_buf, x, x_bytes);
    rc = mu_gpu_dense_probe_ctx(ctx, x_buf, w_buf, out_buf, rows, cols);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, out_bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_dense_bf16_bias_probe(mu_gpu *gpu, const float *x,
                                 const unsigned short *w_bf16,
                                 const unsigned short *bias_bf16,
                                 int rows, int cols, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->dense_bf16_bias_probe) return -1;
    if (!x || !w_bf16 || !bias_bf16 || !out || rows <= 0 || cols <= 0) return -2;

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger x_bytes = (NSUInteger)cols * sizeof(float);
        NSUInteger w_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(unsigned short);
        NSUInteger bias_bytes = (NSUInteger)rows * sizeof(unsigned short);
        NSUInteger out_bytes = (NSUInteger)rows * sizeof(float);

        NSUInteger x_buf_offset = 0;
        id<MTLBuffer> x_buf = mu_scratch_alloc_a(&alloc_ctx, x_bytes, &x_buf_offset);
        if (x_buf) {
            memcpy((char *)[x_buf contents] + x_buf_offset, x, x_bytes);
        } else {
            x_buf = [gpu->device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> w_buf = mu_gpu_get_or_create_buffer(gpu, w_bf16, w_bytes);
        id<MTLBuffer> bias_buf = mu_gpu_get_or_create_buffer(gpu, bias_bf16, bias_bytes);
        NSUInteger out_buf_offset = 0;
        id<MTLBuffer> out_buf = mu_scratch_alloc_b(&alloc_ctx, out_bytes, &out_buf_offset);
        if (!out_buf) {
            out_buf = [gpu->device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        }

        if (!x_buf || !w_buf || !bias_buf || !out_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        bool use_simd = getenv("MU_USE_SIMD") != NULL;
        if (use_simd && gpu->dense_bf16_bias_probe_simd) {
            [encoder setComputePipelineState:gpu->dense_bf16_bias_probe_simd];
            [encoder setBuffer:x_buf offset:x_buf_offset atIndex:0];
            [encoder setBuffer:w_buf offset:0 atIndex:1];
            [encoder setBuffer:bias_buf offset:0 atIndex:2];
            [encoder setBuffer:out_buf offset:out_buf_offset atIndex:3];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:4];

            MTLSize grid = MTLSizeMake(32, (NSUInteger)rows, 1);
            MTLSize threads = MTLSizeMake(32, 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        } else {
            [encoder setComputePipelineState:gpu->dense_bf16_bias_probe];
            [encoder setBuffer:x_buf offset:x_buf_offset atIndex:0];
            [encoder setBuffer:w_buf offset:0 atIndex:1];
            [encoder setBuffer:bias_buf offset:0 atIndex:2];
            [encoder setBuffer:out_buf offset:out_buf_offset atIndex:3];
            [encoder setBytes:&cols length:sizeof(cols) atIndex:4];

            NSUInteger width = gpu->dense_bf16_bias_probe.threadExecutionWidth;
            if (width < 1) width = 1;
            if (width > (NSUInteger)rows) width = (NSUInteger)rows;
            MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
            MTLSize threads = MTLSizeMake(width, 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        }
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_buf_offset, out_bytes);
    }
    return 0;
}

int mu_gpu_dense_f32_bias_probe(mu_gpu *gpu, const float *x,
                                const unsigned short *w_bf16,
                                const unsigned short *bias_bf16,
                                int rows, int cols, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger x_bytes = (NSUInteger)cols * sizeof(float);
    NSUInteger w_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(unsigned short);
    NSUInteger bias_bytes = (NSUInteger)rows * sizeof(unsigned short);
    NSUInteger out_bytes = (NSUInteger)rows * sizeof(float);
    mu_gpu_buf x_buf = mu_gpu_scratch_alloc_a_ctx(ctx, x_bytes);
    mu_gpu_buf w_buf = mu_gpu_get_weight_buf(gpu, w_bf16, w_bytes);
    mu_gpu_buf bias_buf = mu_gpu_get_weight_buf(gpu, bias_bf16, bias_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!x_buf.ptr || !w_buf.ptr || !bias_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(x_buf, x, x_bytes);
    rc = mu_gpu_dense_f32_bias_probe_ctx(ctx, x_buf, w_buf, bias_buf, out_buf, rows, cols);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, out_bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_dense_f32_rows(mu_gpu *gpu, const float *x,
                          const unsigned short *w_bf16,
                          int x_rows, int cols, int out_cols,
                          float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->dense_f32_rows) return -1;
    if (!x || !w_bf16 || !out || x_rows <= 0 || cols <= 0 || out_cols <= 0) return -2;

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger x_bytes = (NSUInteger)x_rows * (NSUInteger)cols * sizeof(float);
        NSUInteger w_bytes = (NSUInteger)out_cols * (NSUInteger)cols * sizeof(unsigned short);
        NSUInteger out_bytes = (NSUInteger)x_rows * (NSUInteger)out_cols * sizeof(float);
        NSUInteger x_buf_offset = 0;
        id<MTLBuffer> x_buf = mu_scratch_alloc_a(&alloc_ctx, x_bytes, &x_buf_offset);
        if (x_buf) {
            memcpy((char *)[x_buf contents] + x_buf_offset, x, x_bytes);
        } else {
            x_buf = [gpu->device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> w_buf = mu_gpu_get_or_create_buffer(gpu, w_bf16, w_bytes);
        NSUInteger out_buf_offset = 0;
        id<MTLBuffer> out_buf = mu_scratch_alloc_b(&alloc_ctx, out_bytes, &out_buf_offset);
        if (!out_buf) {
            out_buf = [gpu->device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        }


        if (!x_buf || !w_buf || !out_buf) return -3;

        bool request_f32_mps = getenv("MU_DENSE_F32_ROWS_MPS") != NULL;
        bool disable_f32_mps = getenv("MU_DENSE_F32_ROWS_NO_MPS") != NULL;
        if ((request_f32_mps || !disable_f32_mps) &&
            mu_gpu_dense_mps_text_shape(cols, out_cols)) {
            return mu_gpu_dense_f32_rows_mps(gpu, x_buf, x_buf_offset,
                                             w_buf, 0,
                                             out_buf, out_buf_offset,
                                             x_rows, cols, out_cols, out);
        }

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->dense_f32_rows];
        [encoder setBuffer:x_buf offset:x_buf_offset atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:out_buf_offset atIndex:2];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
        [encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:4];

        NSUInteger width = gpu->dense_f32_rows.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)out_cols) width = (NSUInteger)out_cols;
        MTLSize grid = MTLSizeMake((NSUInteger)out_cols, (NSUInteger)x_rows, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_buf_offset, out_bytes);
    }
    return 0;
}

int mu_gpu_dense_f32_bias_rows(mu_gpu *gpu, const float *x,
                               const unsigned short *w_bf16,
                               const unsigned short *bias_bf16,
                               int x_rows, int cols, int out_cols,
                               float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->dense_f32_bias_rows) return -1;
    if (!x || !w_bf16 || !bias_bf16 || !out ||
        x_rows <= 0 || cols <= 0 || out_cols <= 0) {
        return -2;
    }

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger x_bytes = (NSUInteger)x_rows * (NSUInteger)cols * sizeof(float);
        NSUInteger w_bytes = (NSUInteger)out_cols * (NSUInteger)cols * sizeof(unsigned short);
        NSUInteger bias_bytes = (NSUInteger)out_cols * sizeof(unsigned short);
        NSUInteger out_bytes = (NSUInteger)x_rows * (NSUInteger)out_cols * sizeof(float);
        NSUInteger x_buf_offset = 0;
        id<MTLBuffer> x_buf = mu_scratch_alloc_a(&alloc_ctx, x_bytes, &x_buf_offset);
        if (x_buf) {
            memcpy((char *)[x_buf contents] + x_buf_offset, x, x_bytes);
        } else {
            x_buf = [gpu->device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> w_buf = mu_gpu_get_or_create_buffer(gpu, w_bf16, w_bytes);
        id<MTLBuffer> bias_buf = mu_gpu_get_or_create_buffer(gpu, bias_bf16, bias_bytes);
        NSUInteger out_buf_offset = 0;
        id<MTLBuffer> out_buf = mu_scratch_alloc_b(&alloc_ctx, out_bytes, &out_buf_offset);
        if (!out_buf) {
            out_buf = [gpu->device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        }


        if (!x_buf || !w_buf || !bias_buf || !out_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->dense_f32_bias_rows];
        [encoder setBuffer:x_buf offset:x_buf_offset atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:bias_buf offset:0 atIndex:2];
        [encoder setBuffer:out_buf offset:out_buf_offset atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:5];

        NSUInteger width = gpu->dense_f32_bias_rows.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)out_cols) width = (NSUInteger)out_cols;
        MTLSize grid = MTLSizeMake((NSUInteger)out_cols, (NSUInteger)x_rows, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_buf_offset, out_bytes);
    }
    return 0;
}

int mu_gpu_dense_bf16_bias_rows(mu_gpu *gpu, const float *x,
                                const unsigned short *w_bf16,
                                const unsigned short *bias_bf16,
                                int x_rows, int cols, int out_cols,
                                float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger x_bytes = (NSUInteger)x_rows * (NSUInteger)cols * sizeof(float);
    NSUInteger w_bytes = (NSUInteger)out_cols * (NSUInteger)cols * sizeof(unsigned short);
    NSUInteger bias_bytes = (NSUInteger)out_cols * sizeof(unsigned short);
    NSUInteger out_bytes = (NSUInteger)x_rows * (NSUInteger)out_cols * sizeof(float);
    mu_gpu_buf x_buf = mu_gpu_scratch_alloc_a_ctx(ctx, x_bytes);
    mu_gpu_buf w_buf = mu_gpu_get_weight_buf(gpu, w_bf16, w_bytes);
    mu_gpu_buf bias_buf = mu_gpu_get_weight_buf(gpu, bias_bf16, bias_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!x_buf.ptr || !w_buf.ptr || !bias_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(x_buf, x, x_bytes);
    rc = mu_gpu_dense_bf16_bias_rows_ctx(ctx, x_buf, w_buf, bias_buf, x_rows, cols, out_cols, out_buf);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, out_bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_rmsnorm_probe(mu_gpu *gpu, const float *x, const float *weight,
                         int n, float eps, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->rmsnorm_probe) return -1;
    if (!x || !weight || !out || n <= 0) return -2;

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger bytes = (NSUInteger)n * sizeof(float);
        NSUInteger x_buf_offset = 0;
        id<MTLBuffer> x_buf = mu_scratch_alloc_a(&alloc_ctx, bytes, &x_buf_offset);
        if (x_buf) {
            memcpy((char *)[x_buf contents] + x_buf_offset, x, bytes);
        } else {
            x_buf = [gpu->device newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger w_buf_offset = 0;
        id<MTLBuffer> w_buf = mu_scratch_alloc_a(&alloc_ctx, bytes, &w_buf_offset);
        if (w_buf) {
            memcpy((char *)[w_buf contents] + w_buf_offset, weight, bytes);
        } else {
            w_buf = [gpu->device newBufferWithBytes:weight length:bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger out_buf_offset = 0;
        id<MTLBuffer> out_buf = mu_scratch_alloc_b(&alloc_ctx, bytes, &out_buf_offset);
        if (!out_buf) {
            out_buf = [gpu->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        }


        if (!x_buf || !w_buf || !out_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->rmsnorm_probe];
        [encoder setBuffer:x_buf offset:x_buf_offset atIndex:0];
        [encoder setBuffer:w_buf offset:w_buf_offset atIndex:1];
        [encoder setBuffer:out_buf offset:out_buf_offset atIndex:2];
        [encoder setBytes:&n length:sizeof(n) atIndex:3];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:4];

        NSUInteger width = gpu->rmsnorm_probe.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)n) width = (NSUInteger)n;
        MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_buf_offset, bytes);
    }
    return 0;
}

int mu_gpu_rmsnorm_bf16_probe(mu_gpu *gpu, const float *x,
                              const unsigned short *weight_bf16,
                              int n, float eps, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger x_bytes = (NSUInteger)n * sizeof(float);
    NSUInteger w_bytes = (NSUInteger)n * sizeof(unsigned short);
    NSUInteger out_bytes = (NSUInteger)n * sizeof(float);
    mu_gpu_buf x_buf = mu_gpu_scratch_alloc_a_ctx(ctx, x_bytes);
    mu_gpu_buf w_buf = mu_gpu_get_weight_buf(gpu, weight_bf16, w_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!x_buf.ptr || !w_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(x_buf, x, x_bytes);
    rc = mu_gpu_rmsnorm_bf16_probe_ctx(ctx, x_buf, w_buf, out_buf, n, eps);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, out_bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_rmsnorm_bf16_rows(mu_gpu *gpu, const float *x,
                             const unsigned short *weight_bf16,
                             int rows, int cols, float eps, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->rmsnorm_bf16_rows) return -1;
    if (!x || !weight_bf16 || !out || rows <= 0 || cols <= 0) return -2;

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger x_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
        NSUInteger w_bytes = (NSUInteger)cols * sizeof(unsigned short);
        NSUInteger x_buf_offset = 0;
        id<MTLBuffer> x_buf = mu_scratch_alloc_a(&alloc_ctx, x_bytes, &x_buf_offset);
        if (x_buf) {
            memcpy((char *)[x_buf contents] + x_buf_offset, x, x_bytes);
        } else {
            x_buf = [gpu->device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> w_buf = mu_gpu_get_or_create_buffer(gpu, weight_bf16, w_bytes);
        NSUInteger out_buf_offset = 0;
        id<MTLBuffer> out_buf = mu_scratch_alloc_b(&alloc_ctx, x_bytes, &out_buf_offset);
        if (!out_buf) {
            out_buf = [gpu->device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
        }


        if (!x_buf || !w_buf || !out_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->rmsnorm_bf16_rows];
        [encoder setBuffer:x_buf offset:x_buf_offset atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:out_buf_offset atIndex:2];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:4];

        NSUInteger width = gpu->rmsnorm_bf16_rows.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)cols) width = (NSUInteger)cols;
        MTLSize grid = MTLSizeMake((NSUInteger)cols, (NSUInteger)rows, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_buf_offset, x_bytes);
    }
    return 0;
}

int mu_gpu_layernorm_bf16_probe(mu_gpu *gpu, const float *x,
                                const unsigned short *weight_bf16,
                                const unsigned short *bias_bf16,
                                int n, float eps, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->layernorm_bf16_probe) return -1;
    if (!x || !weight_bf16 || !bias_bf16 || !out || n <= 0) return -2;

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger x_bytes = (NSUInteger)n * sizeof(float);
        NSUInteger bf16_bytes = (NSUInteger)n * sizeof(unsigned short);
        NSUInteger x_buf_offset = 0;
        id<MTLBuffer> x_buf = mu_scratch_alloc_a(&alloc_ctx, x_bytes, &x_buf_offset);
        if (x_buf) {
            memcpy((char *)[x_buf contents] + x_buf_offset, x, x_bytes);
        } else {
            x_buf = [gpu->device newBufferWithBytes:x length:x_bytes options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> w_buf = mu_gpu_get_or_create_buffer(gpu, weight_bf16, bf16_bytes);
        id<MTLBuffer> b_buf = mu_gpu_get_or_create_buffer(gpu, bias_bf16, bf16_bytes);
        NSUInteger out_buf_offset = 0;
        id<MTLBuffer> out_buf = mu_scratch_alloc_b(&alloc_ctx, x_bytes, &out_buf_offset);
        if (!out_buf) {
            out_buf = [gpu->device newBufferWithLength:x_bytes options:MTLResourceStorageModeShared];
        }


        if (!x_buf || !w_buf || !b_buf || !out_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->layernorm_bf16_probe];
        [encoder setBuffer:x_buf offset:x_buf_offset atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:b_buf offset:0 atIndex:2];
        [encoder setBuffer:out_buf offset:out_buf_offset atIndex:3];
        [encoder setBytes:&n length:sizeof(n) atIndex:4];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:5];

        NSUInteger width = gpu->layernorm_bf16_probe.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)n) width = (NSUInteger)n;
        MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_buf_offset, x_bytes);
    }
    return 0;
}

int mu_gpu_layernorm_bf16_rows(mu_gpu *gpu, const float *x,
                               const unsigned short *weight_bf16,
                               const unsigned short *bias_bf16,
                               int rows, int cols, float eps, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger x_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
    NSUInteger bf16_bytes = (NSUInteger)cols * sizeof(unsigned short);
    NSUInteger out_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
    mu_gpu_buf x_buf = mu_gpu_scratch_alloc_a_ctx(ctx, x_bytes);
    mu_gpu_buf w_buf = mu_gpu_get_weight_buf(gpu, weight_bf16, bf16_bytes);
    mu_gpu_buf bias_buf = mu_gpu_get_weight_buf(gpu, bias_bf16, bf16_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!x_buf.ptr || !w_buf.ptr || !bias_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(x_buf, x, x_bytes);
    rc = mu_gpu_layernorm_bf16_rows_ctx(ctx, x_buf, w_buf, bias_buf, rows, cols, eps, out_buf);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, out_bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_vision_attn_concat_probe(mu_gpu *gpu, const float *q0,
                                     const float *kv, const float *rotary,
                                     int rows, int token_index, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger q0_bytes = 1280u * sizeof(float);
    NSUInteger kv_bytes = (NSUInteger)rows * 2560u * sizeof(float);
    NSUInteger out_bytes = 1280u * sizeof(float);
    mu_gpu_buf q0_buf = mu_gpu_scratch_alloc_a_ctx(ctx, q0_bytes);
    mu_gpu_buf kv_buf = mu_gpu_scratch_alloc_a_ctx(ctx, kv_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!q0_buf.ptr || !kv_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(q0_buf, q0, q0_bytes);
    mu_gpu_buf_copy_to(kv_buf, kv, kv_bytes);
    rc = mu_gpu_vision_attn_concat_probe_ctx(ctx, q0_buf, kv_buf, rotary, rows, token_index, out_buf);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, out_bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_vision_attn_rows(mu_gpu *gpu, const float *q,
                            const float *kv, const float *rotary,
                            int rows, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger q_bytes = (NSUInteger)rows * 1280u * sizeof(float);
    NSUInteger kv_bytes = (NSUInteger)rows * 2560u * sizeof(float);
    NSUInteger out_bytes = (NSUInteger)rows * 1280u * sizeof(float);
    mu_gpu_buf q_buf = mu_gpu_scratch_alloc_a_ctx(ctx, q_bytes);
    mu_gpu_buf kv_buf = mu_gpu_scratch_alloc_a_ctx(ctx, kv_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!q_buf.ptr || !kv_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(q_buf, q, q_bytes);
    mu_gpu_buf_copy_to(kv_buf, kv, kv_bytes);
    rc = mu_gpu_vision_attn_rows_ctx(ctx, q_buf, kv_buf, rotary, rows, out_buf);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, out_bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_text_attn_token0(mu_gpu *gpu, const float *v, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->text_attn_token0) return -1;
    if (!v || !out) return -2;

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger v_bytes = 128u * sizeof(float);
        NSUInteger out_bytes = 896u * sizeof(float);
        NSUInteger v_buf_offset = 0;
        id<MTLBuffer> v_buf = mu_scratch_alloc_a(&alloc_ctx, v_bytes, &v_buf_offset);
        if (v_buf) {
            memcpy((char *)[v_buf contents] + v_buf_offset, v, v_bytes);
        } else {
            v_buf = [gpu->device newBufferWithBytes:v length:v_bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger out_buf_offset = 0;
        id<MTLBuffer> out_buf = mu_scratch_alloc_b(&alloc_ctx, out_bytes, &out_buf_offset);
        if (!out_buf) {
            out_buf = [gpu->device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        }
        if (!v_buf || !out_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->text_attn_token0];
        [encoder setBuffer:v_buf offset:v_buf_offset atIndex:0];
        [encoder setBuffer:out_buf offset:out_buf_offset atIndex:1];

        NSUInteger width = gpu->text_attn_token0.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > 896u) width = 896u;
        MTLSize grid = MTLSizeMake(896u, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_buf_offset, out_bytes);
    }
    return 0;
}

int mu_gpu_text_attn_seq(mu_gpu *gpu, const float *q, const float *k,
                         const float *v, int seq, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->text_attn_seq) return -1;
    if (!q || !k || !v || !out || seq <= 0) return -2;

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger q_bytes = (NSUInteger)seq * 896u * sizeof(float);
        NSUInteger kv_bytes = (NSUInteger)seq * 128u * sizeof(float);
        NSUInteger out_bytes = q_bytes;
        NSUInteger q_buf_offset = 0;
        id<MTLBuffer> q_buf = mu_scratch_alloc_a(&alloc_ctx, q_bytes, &q_buf_offset);
        if (q_buf) {
            memcpy((char *)[q_buf contents] + q_buf_offset, q, q_bytes);
        } else {
            q_buf = [gpu->device newBufferWithBytes:q length:q_bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger k_buf_offset = 0;
        id<MTLBuffer> k_buf = mu_scratch_alloc_a(&alloc_ctx, kv_bytes, &k_buf_offset);
        if (k_buf) {
            memcpy((char *)[k_buf contents] + k_buf_offset, k, kv_bytes);
        } else {
            k_buf = [gpu->device newBufferWithBytes:k length:kv_bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger v_buf_offset = 0;
        id<MTLBuffer> v_buf = mu_scratch_alloc_a(&alloc_ctx, kv_bytes, &v_buf_offset);
        if (v_buf) {
            memcpy((char *)[v_buf contents] + v_buf_offset, v, kv_bytes);
        } else {
            v_buf = [gpu->device newBufferWithBytes:v length:kv_bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger out_buf_offset = 0;
        id<MTLBuffer> out_buf = mu_scratch_alloc_b(&alloc_ctx, out_bytes, &out_buf_offset);
        if (!out_buf) {
            out_buf = [gpu->device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        }

        if (!q_buf || !k_buf || !v_buf || !out_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->text_attn_seq];
        [encoder setBuffer:q_buf offset:q_buf_offset atIndex:0];
        [encoder setBuffer:k_buf offset:k_buf_offset atIndex:1];
        [encoder setBuffer:v_buf offset:v_buf_offset atIndex:2];
        [encoder setBuffer:out_buf offset:out_buf_offset atIndex:3];
        [encoder setBytes:&seq length:sizeof(seq) atIndex:4];

        NSUInteger total = (NSUInteger)seq * 896u;
        NSUInteger width = gpu->text_attn_seq.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > total) width = total;
        MTLSize grid = MTLSizeMake(total, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_buf_offset, out_bytes);
    }
    return 0;
}

int mu_gpu_text_attn_seq_pos(mu_gpu *gpu, const float *q, const float *k,
                             const float *v, const int *position_ids,
                             int seq, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->text_attn_seq_pos) return -1;
    if (!q || !k || !v || !position_ids || !out || seq <= 0) return -2;

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger q_bytes = (NSUInteger)seq * 896u * sizeof(float);
        NSUInteger kv_bytes = (NSUInteger)seq * 128u * sizeof(float);
        NSUInteger pos_bytes = (NSUInteger)seq * 3u * sizeof(int);
        NSUInteger out_bytes = q_bytes;
        NSUInteger q_buf_offset = 0;
        id<MTLBuffer> q_buf = mu_scratch_alloc_a(&alloc_ctx, q_bytes, &q_buf_offset);
        if (q_buf) {
            memcpy((char *)[q_buf contents] + q_buf_offset, q, q_bytes);
        } else {
            q_buf = [gpu->device newBufferWithBytes:q length:q_bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger k_buf_offset = 0;
        id<MTLBuffer> k_buf = mu_scratch_alloc_a(&alloc_ctx, kv_bytes, &k_buf_offset);
        if (k_buf) {
            memcpy((char *)[k_buf contents] + k_buf_offset, k, kv_bytes);
        } else {
            k_buf = [gpu->device newBufferWithBytes:k length:kv_bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger v_buf_offset = 0;
        id<MTLBuffer> v_buf = mu_scratch_alloc_a(&alloc_ctx, kv_bytes, &v_buf_offset);
        if (v_buf) {
            memcpy((char *)[v_buf contents] + v_buf_offset, v, kv_bytes);
        } else {
            v_buf = [gpu->device newBufferWithBytes:v length:kv_bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger pos_buf_offset = 0;
        id<MTLBuffer> pos_buf = mu_scratch_alloc_a(&alloc_ctx, pos_bytes, &pos_buf_offset);
        if (pos_buf) {
            memcpy((char *)[pos_buf contents] + pos_buf_offset, position_ids, pos_bytes);
        } else {
            pos_buf = [gpu->device newBufferWithBytes:position_ids length:pos_bytes options:MTLResourceStorageModeShared];
        }
        NSUInteger out_buf_offset = 0;
        id<MTLBuffer> out_buf = mu_scratch_alloc_b(&alloc_ctx, out_bytes, &out_buf_offset);
        if (!out_buf) {
            out_buf = [gpu->device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        }

        if (!q_buf || !k_buf || !v_buf || !pos_buf || !out_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->text_attn_seq_pos];
        [encoder setBuffer:q_buf offset:q_buf_offset atIndex:0];
        [encoder setBuffer:k_buf offset:k_buf_offset atIndex:1];
        [encoder setBuffer:v_buf offset:v_buf_offset atIndex:2];
        [encoder setBuffer:pos_buf offset:pos_buf_offset atIndex:3];
        [encoder setBuffer:out_buf offset:out_buf_offset atIndex:4];
        [encoder setBytes:&seq length:sizeof(seq) atIndex:5];

        NSUInteger width = gpu->text_attn_seq_pos.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)seq) width = (NSUInteger)seq;
        MTLSize grid = MTLSizeMake((NSUInteger)seq, 14, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_buf_offset, out_bytes);
    }
    return 0;
}

int mu_gpu_text_attn_cached(mu_gpu *gpu, const float *q,
                            const float *k_cache, const float *v_cache,
                            int cache_len, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger q_bytes = 896u * sizeof(float);
    NSUInteger out_bytes = 896u * sizeof(float);
    mu_gpu_buf q_buf = mu_gpu_scratch_alloc_a_ctx(ctx, q_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!q_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(q_buf, q, q_bytes);
    rc = mu_gpu_text_attn_cached_ctx(ctx, q_buf, k_cache, v_cache, cache_len, out_buf);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, out_bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_add_f32(mu_gpu *gpu, const float *a, const float *b,
                   int n, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger bytes = (NSUInteger)n * sizeof(float);
    mu_gpu_buf a_buf = mu_gpu_scratch_alloc_a_ctx(ctx, bytes);
    mu_gpu_buf b_buf = mu_gpu_scratch_alloc_a_ctx(ctx, bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, bytes);
    if (!a_buf.ptr || !b_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(a_buf, a, bytes);
    mu_gpu_buf_copy_to(b_buf, b, bytes);
    rc = mu_gpu_add_f32_ctx(ctx, a_buf, b_buf, out_buf, n);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_silu_mul_f32(mu_gpu *gpu, const float *gate, const float *up,
                        int n, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger bytes = (NSUInteger)n * sizeof(float);
    mu_gpu_buf gate_buf = mu_gpu_scratch_alloc_a_ctx(ctx, bytes);
    mu_gpu_buf up_buf = mu_gpu_scratch_alloc_a_ctx(ctx, bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, bytes);
    if (!gate_buf.ptr || !up_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(gate_buf, gate, bytes);
    mu_gpu_buf_copy_to(up_buf, up, bytes);
    rc = mu_gpu_silu_mul_f32_ctx(ctx, gate_buf, up_buf, out_buf, n);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_vision_add_bf16(mu_gpu *gpu, const float *a, const float *b,
                            int n, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger bytes = (NSUInteger)n * sizeof(float);
    mu_gpu_buf a_buf = mu_gpu_scratch_alloc_a_ctx(ctx, bytes);
    mu_gpu_buf b_buf = mu_gpu_scratch_alloc_a_ctx(ctx, bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, bytes);
    if (!a_buf.ptr || !b_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(a_buf, a, bytes);
    mu_gpu_buf_copy_to(b_buf, b, bytes);
    rc = mu_gpu_vision_add_bf16_ctx(ctx, a_buf, b_buf, n, out_buf);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_vision_quick_gelu_bf16(mu_gpu *gpu, const float *x,
                                   int n, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger bytes = (NSUInteger)n * sizeof(float);
    mu_gpu_buf x_buf = mu_gpu_scratch_alloc_a_ctx(ctx, bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, bytes);
    if (!x_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(x_buf, x, bytes);
    rc = mu_gpu_vision_quick_gelu_bf16_ctx(ctx, x_buf, n, out_buf);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_vision_gelu_bf16(mu_gpu *gpu, const float *x,
                                 int n, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger bytes = (NSUInteger)n * sizeof(float);
    mu_gpu_buf x_buf = mu_gpu_scratch_alloc_a_ctx(ctx, bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, bytes);
    if (!x_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(x_buf, x, bytes);
    rc = mu_gpu_vision_gelu_bf16_ctx(ctx, x_buf, n, out_buf);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_vision_merge4(mu_gpu *gpu, const float *hidden,
                         int rows, float *out) {
    mu_gpu_cmd_ctx *ctx = NULL;
    int rc = mu_gpu_cmd_begin(gpu, &ctx);
    if (rc != 0) return rc;
    NSUInteger in_bytes = (NSUInteger)rows * 1280u * sizeof(float);
    NSUInteger out_bytes = ((NSUInteger)rows / 4u) * 5120u * sizeof(float);
    mu_gpu_buf hidden_buf = mu_gpu_scratch_alloc_a_ctx(ctx, in_bytes);
    mu_gpu_buf out_buf = mu_gpu_scratch_alloc_b_ctx(ctx, out_bytes);
    if (!hidden_buf.ptr || !out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -3; }
    mu_gpu_buf_copy_to(hidden_buf, hidden, in_bytes);
    rc = mu_gpu_vision_merge4_ctx(ctx, hidden_buf, rows, out_buf);
    if (rc == 0) {
        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) mu_gpu_buf_copy_from(out, out_buf, out_bytes);
    } else { mu_gpu_cmd_discard(ctx); }
    return rc;
}

int mu_gpu_vision_encode(mu_gpu *gpu, void *engine,
                         const float *patch_embeds,
                         int rows, int cols,
                         const float *rotary,
                         int rotary_rows, int rotary_cols,
                         float *out, int out_rows, int out_cols) {
    if (!gpu || !engine || !patch_embeds || !rotary || !out || rows <= 0 || cols != 1280) return -1;
    (void)rotary_rows;
    (void)rotary_cols;
    (void)out_rows;
    (void)out_cols;

    @autoreleasepool {
        mu_gpu_cmd_ctx *ctx = NULL;
        int rc = mu_gpu_cmd_begin(gpu, &ctx);
        if (rc != 0) return rc;
        mu_gpu_cmd_set_label(ctx, "vision_encode_vit_and_merger");
        bool profile_split = getenv("MU_VISION_PROFILE_SPLIT") != NULL;
        char profile_label[96];

        unsigned long embed_bytes = (unsigned long)rows * 1280u * sizeof(float);
        unsigned long kv_bytes = (unsigned long)rows * 2560u * sizeof(float);
        unsigned long mlp_bytes = (unsigned long)rows * 5120u * sizeof(float);

        if (getenv("MU_METAL_DEBUG")) {
            fprintf(stderr, "DEBUG: mu_gpu_vision_encode rows = %d, cols = %d, embed_bytes = %lu, kv_bytes = %lu, mlp_bytes = %lu\n", rows, cols, embed_bytes, kv_bytes, mlp_bytes);
        }

        mu_gpu_buf ping_buf = mu_gpu_scratch_alloc_a_ctx(ctx, embed_bytes);
        mu_gpu_buf pong_buf = mu_gpu_scratch_alloc_b_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_normed = mu_gpu_scratch_alloc_a_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_q = mu_gpu_scratch_alloc_a_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_kv = mu_gpu_scratch_alloc_a_ctx(ctx, kv_bytes);
        mu_gpu_buf temp_attn = mu_gpu_scratch_alloc_b_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_proj = mu_gpu_scratch_alloc_b_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_res1 = mu_gpu_scratch_alloc_b_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_norm2 = mu_gpu_scratch_alloc_a_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_fc1 = mu_gpu_scratch_alloc_b_ctx(ctx, mlp_bytes);
        mu_gpu_buf temp_fc1_act = mu_gpu_scratch_alloc_a_ctx(ctx, mlp_bytes);

        if (!ping_buf.ptr || !pong_buf.ptr || !temp_normed.ptr || !temp_q.ptr || !temp_kv.ptr ||
            !temp_attn.ptr || !temp_proj.ptr || !temp_res1.ptr || !temp_norm2.ptr ||
            !temp_fc1.ptr || !temp_fc1_act.ptr) {
            mu_gpu_cmd_discard(ctx);
            return -2;
        }

        mu_gpu_buf_copy_to(ping_buf, patch_embeds, embed_bytes);

        mu_gpu_buf current_in = ping_buf;
        mu_gpu_buf current_out = pong_buf;

        int layers = mu_engine_vision_layers((const mu_engine *)engine);
        bool shape_profile = getenv("MU_VISION_ATTN_SHAPE_PROFILE") != NULL;

        NSUInteger base_offset_a = ctx->alloc.offset_a;
        NSUInteger base_offset_b = ctx->alloc.offset_b;

        for (int layer = 0; layer < layers; layer++) {
            ctx->alloc.offset_a = base_offset_a;
            ctx->alloc.offset_b = base_offset_b;
            if (profile_split) {
                snprintf(profile_label, sizeof(profile_label), "vision_profile_norm1_l%02d", layer);
                mu_gpu_cmd_set_label(ctx, profile_label);
            }
            const unsigned short *norm1_w = mu_engine_get_vision_block_tensor(engine, layer, "norm1.weight", 1, 1280, 0);
            const unsigned short *norm1_b = mu_engine_get_vision_block_tensor(engine, layer, "norm1.bias", 1, 1280, 0);
            const unsigned short *qkv_w = mu_engine_get_vision_block_tensor(engine, layer, "attn.qkv.weight", 2, 3840, 1280);
            const unsigned short *qkv_b = mu_engine_get_vision_block_tensor(engine, layer, "attn.qkv.bias", 1, 3840, 0);
            const unsigned short *proj_w = mu_engine_get_vision_block_tensor(engine, layer, "attn.proj.weight", 2, 1280, 1280);
            const unsigned short *proj_b = mu_engine_get_vision_block_tensor(engine, layer, "attn.proj.bias", 1, 1280, 0);
            const unsigned short *norm2_w = mu_engine_get_vision_block_tensor(engine, layer, "norm2.weight", 1, 1280, 0);
            const unsigned short *norm2_b = mu_engine_get_vision_block_tensor(engine, layer, "norm2.bias", 1, 1280, 0);
            const unsigned short *fc1_w = mu_engine_get_vision_block_tensor(engine, layer, "mlp.fc1.weight", 2, 5120, 1280);
            const unsigned short *fc1_b = mu_engine_get_vision_block_tensor(engine, layer, "mlp.fc1.bias", 1, 5120, 0);
            const unsigned short *fc2_w = mu_engine_get_vision_block_tensor(engine, layer, "mlp.fc2.weight", 2, 1280, 5120);
            const unsigned short *fc2_b = mu_engine_get_vision_block_tensor(engine, layer, "mlp.fc2.bias", 1, 1280, 0);

            if (!norm1_w || !norm1_b || !qkv_w || !qkv_b || !proj_w || !proj_b ||
                !norm2_w || !norm2_b || !fc1_w || !fc1_b || !fc2_w || !fc2_b) {
                mu_gpu_cmd_discard(ctx);
                return -3;
            }

            mu_gpu_buf norm1_w_buf = mu_gpu_get_weight_buf(gpu, norm1_w, 1280 * sizeof(unsigned short));
            mu_gpu_buf norm1_b_buf = mu_gpu_get_weight_buf(gpu, norm1_b, 1280 * sizeof(unsigned short));
            mu_gpu_buf qkv_w_buf = mu_gpu_get_weight_buf(gpu, qkv_w, 3840 * 1280 * sizeof(unsigned short));
            mu_gpu_buf qkv_b_buf = mu_gpu_get_weight_buf(gpu, qkv_b, 3840 * sizeof(unsigned short));
            mu_gpu_buf proj_w_buf = mu_gpu_get_weight_buf(gpu, proj_w, 1280 * 1280 * sizeof(unsigned short));
            mu_gpu_buf proj_b_buf = mu_gpu_get_weight_buf(gpu, proj_b, 1280 * sizeof(unsigned short));
            mu_gpu_buf norm2_w_buf = mu_gpu_get_weight_buf(gpu, norm2_w, 1280 * sizeof(unsigned short));
            mu_gpu_buf norm2_b_buf = mu_gpu_get_weight_buf(gpu, norm2_b, 1280 * sizeof(unsigned short));
            mu_gpu_buf fc1_w_buf = mu_gpu_get_weight_buf(gpu, fc1_w, 5120 * 1280 * sizeof(unsigned short));
            mu_gpu_buf fc1_b_buf = mu_gpu_get_weight_buf(gpu, fc1_b, 5120 * sizeof(unsigned short));
            mu_gpu_buf fc2_w_buf = mu_gpu_get_weight_buf(gpu, fc2_w, 1280 * 5120 * sizeof(unsigned short));
            mu_gpu_buf fc2_b_buf = mu_gpu_get_weight_buf(gpu, fc2_b, 1280 * sizeof(unsigned short));

            if (!norm1_w_buf.ptr || !norm1_b_buf.ptr || !qkv_w_buf.ptr || !qkv_b_buf.ptr ||
                !proj_w_buf.ptr || !proj_b_buf.ptr || !norm2_w_buf.ptr || !norm2_b_buf.ptr ||
                !fc1_w_buf.ptr || !fc1_b_buf.ptr || !fc2_w_buf.ptr || !fc2_b_buf.ptr) {
                mu_gpu_cmd_discard(ctx);
                return -4;
            }

            rc = mu_gpu_layernorm_bf16_rows_ctx(ctx, current_in, norm1_w_buf, norm1_b_buf, rows, 1280, 1e-6f, temp_normed);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
            if (profile_split) {
                snprintf(profile_label, sizeof(profile_label), "vision_profile_qkv_l%02d", layer);
                rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_norm1", profile_label);
                if (rc != 0) return rc;
            }

            bool request_mps = getenv("MU_DENSE_ROWS_MPS") != NULL;
            bool use_simdgroup = mu_gpu_dense_mps_shape(1280, 3840) &&
                                 gpu->dense_bf16_bias_rows_simdgroup_qkv &&
                                 getenv("MU_DENSE_ROWS_NO_SIMDGROUP") == NULL &&
                                 !request_mps;

            if (use_simdgroup) {
                [ctx->encoder setComputePipelineState:gpu->dense_bf16_bias_rows_simdgroup_qkv];
                [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)temp_normed.ptr offset:temp_normed.offset atIndex:0];
                [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)qkv_w_buf.ptr offset:qkv_w_buf.offset atIndex:1];
                [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)qkv_b_buf.ptr offset:qkv_b_buf.offset atIndex:2];
                [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)temp_q.ptr offset:temp_q.offset atIndex:3];
                [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)temp_kv.ptr offset:temp_kv.offset atIndex:4];
                int cols = 1280;
                int out_cols = 3840;
                [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:5];
                [ctx->encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:6];
                [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:7];

                MTLSize grid = MTLSizeMake(((NSUInteger)3840 + 31u) / 32u, ((NSUInteger)rows + 7u) / 8u, 1);
                MTLSize threads = MTLSizeMake(32, 1, 1);
                [ctx->encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
            } else {
                rc = mu_gpu_dense_bf16_bias_rows_ctx(ctx, temp_normed, qkv_w_buf, qkv_b_buf, rows, 1280, 1280, temp_q);
                if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

                mu_gpu_buf kv_w_buf = mu_gpu_get_weight_buf(gpu, qkv_w + (size_t)1280 * 1280, 2560 * 1280 * sizeof(unsigned short));
                mu_gpu_buf kv_b_buf = mu_gpu_get_weight_buf(gpu, qkv_b + 1280, 2560 * sizeof(unsigned short));
                if (!kv_w_buf.ptr || !kv_b_buf.ptr) { mu_gpu_cmd_discard(ctx); return -5; }

                rc = mu_gpu_dense_bf16_bias_rows_ctx(ctx, temp_normed, kv_w_buf, kv_b_buf, rows, 1280, 2560, temp_kv);
                if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
            }
            if (profile_split) {
                snprintf(profile_label, sizeof(profile_label), "vision_profile_attention_l%02d", layer);
                rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_qkv", profile_label);
                if (rc != 0) return rc;
            }

            bool request_mpsgraph = getenv("MU_VISION_ATTN_MPSGRAPH") != NULL;
            bool disable_mpsgraph = getenv("MU_VISION_ATTN_NO_MPSGRAPH") != NULL;
            bool request_legacy_attention =
                getenv("MU_VISION_ATTN_FLASH_K16") != NULL ||
                getenv("MU_VISION_ATTN_NO_FLASH") != NULL ||
                getenv("MU_VISION_ATTN_ONLINE") != NULL;
            bool use_mpsgraph_attention =
                !disable_mpsgraph &&
                (request_mpsgraph || !request_legacy_attention) &&
                gpu->vision_attn_pack_qkv_mpsgraph &&
                gpu->vision_attn_copy_mpsgraph;

            if (rows <= 16) {
                for (int r = 0; rc == 0 && r < rows; r++) {
                    mu_gpu_buf q_r = temp_q;
                    q_r.offset += (size_t)r * 1280u * sizeof(float);
                    mu_gpu_buf attn_r = temp_attn;
                    attn_r.offset += (size_t)r * 1280u * sizeof(float);
                    rc = mu_gpu_vision_attn_concat_probe_ctx(ctx, q_r, temp_kv, rotary, rows, r, attn_r);
                }
            } else if (use_mpsgraph_attention) {
                if (shape_profile) {
                    fprintf(stderr, "mu_profile stage=vision_attn_shape layer=%d rows=%d\n",
                            layer, rows);
                }
                rc = mu_gpu_vision_attn_rows_mpsgraph_stage(gpu, &ctx, temp_q, temp_kv,
                                                            rotary, rows, temp_attn,
                                                            shape_profile);
            } else {
                if (shape_profile) {
                    fprintf(stderr, "mu_profile stage=vision_attn_shape layer=%d rows=%d\n",
                            layer, rows);
                }
                rc = mu_gpu_vision_attn_rows_ctx(ctx, temp_q, temp_kv, rotary, rows, temp_attn);
            }
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
            if (profile_split) {
                snprintf(profile_label, sizeof(profile_label), "vision_profile_proj_l%02d", layer);
                rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_attention", profile_label);
                if (rc != 0) return rc;
            }

            rc = mu_gpu_dense_bf16_bias_rows_ctx(ctx, temp_attn, proj_w_buf, proj_b_buf, rows, 1280, 1280, temp_proj);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
            if (profile_split) {
                snprintf(profile_label, sizeof(profile_label), "vision_profile_residual1_l%02d", layer);
                rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_proj", profile_label);
                if (rc != 0) return rc;
            }

            rc = mu_gpu_vision_add_bf16_ctx(ctx, current_in, temp_proj, rows * 1280, temp_res1);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
            if (profile_split) {
                snprintf(profile_label, sizeof(profile_label), "vision_profile_norm2_l%02d", layer);
                rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_residual1", profile_label);
                if (rc != 0) return rc;
            }

            rc = mu_gpu_layernorm_bf16_rows_ctx(ctx, temp_res1, norm2_w_buf, norm2_b_buf, rows, 1280, 1e-6f, temp_norm2);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
            if (profile_split) {
                snprintf(profile_label, sizeof(profile_label), "vision_profile_fc1_gelu_l%02d", layer);
                rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_norm2", profile_label);
                if (rc != 0) return rc;
            }

            rc = mu_gpu_dense_bf16_bias_rows_quick_gelu_ctx(ctx, temp_norm2, fc1_w_buf, fc1_b_buf, rows, 1280, 5120, temp_fc1_act);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
            if (profile_split) {
                snprintf(profile_label, sizeof(profile_label), "vision_profile_fc2_l%02d", layer);
                rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_fc1_gelu", profile_label);
                if (rc != 0) return rc;
            }

            rc = mu_gpu_dense_bf16_bias_rows_ctx(ctx, temp_fc1_act, fc2_w_buf, fc2_b_buf, rows, 5120, 1280, temp_proj);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
            if (profile_split) {
                snprintf(profile_label, sizeof(profile_label), "vision_profile_residual2_l%02d", layer);
                rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_fc2", profile_label);
                if (rc != 0) return rc;
            }

            rc = mu_gpu_vision_add_bf16_ctx(ctx, temp_res1, temp_proj, rows * 1280, current_out);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
            if (profile_split) {
                rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_residual2", "");
                if (rc != 0) return rc;
            }

            // Swap ping-pong
            mu_gpu_buf tmp = current_in;
            current_in = current_out;
            current_out = tmp;
        }

        ctx->alloc.offset_a = base_offset_a;
        ctx->alloc.offset_b = base_offset_b;

        // --- Merger ---
        const unsigned short *ln_w = mu_engine_get_vision_merger_tensor(engine, "visual.merger.ln_q.weight", 1, 1280, 0);
        const unsigned short *ln_b = mu_engine_get_vision_merger_tensor(engine, "visual.merger.ln_q.bias", 1, 1280, 0);
        const unsigned short *fc0_w = mu_engine_get_vision_merger_tensor(engine, "visual.merger.mlp.0.weight", 2, 5120, 5120);
        const unsigned short *fc0_b = mu_engine_get_vision_merger_tensor(engine, "visual.merger.mlp.0.bias", 1, 5120, 0);
        const unsigned short *fc2_w = mu_engine_get_vision_merger_tensor(engine, "visual.merger.mlp.2.weight", 2, 896, 5120);
        const unsigned short *fc2_b = mu_engine_get_vision_merger_tensor(engine, "visual.merger.mlp.2.bias", 1, 896, 0);

        if (!ln_w || !ln_b || !fc0_w || !fc0_b || !fc2_w || !fc2_b) {
            mu_gpu_cmd_discard(ctx);
            return -6;
        }

        mu_gpu_buf ln_w_buf = mu_gpu_get_weight_buf(gpu, ln_w, 1280 * sizeof(unsigned short));
        mu_gpu_buf ln_b_buf = mu_gpu_get_weight_buf(gpu, ln_b, 1280 * sizeof(unsigned short));
        mu_gpu_buf fc0_w_buf = mu_gpu_get_weight_buf(gpu, fc0_w, 5120 * 5120 * sizeof(unsigned short));
        mu_gpu_buf fc0_b_buf = mu_gpu_get_weight_buf(gpu, fc0_b, 5120 * sizeof(unsigned short));
        mu_gpu_buf fc2_w_buf = mu_gpu_get_weight_buf(gpu, fc2_w, 896 * 5120 * sizeof(unsigned short));
        mu_gpu_buf fc2_b_buf = mu_gpu_get_weight_buf(gpu, fc2_b, 896 * sizeof(unsigned short));

        if (!ln_w_buf.ptr || !ln_b_buf.ptr || !fc0_w_buf.ptr || !fc0_b_buf.ptr || !fc2_w_buf.ptr || !fc2_b_buf.ptr) {
            mu_gpu_cmd_discard(ctx);
            return -7;
        }

        int groups = rows / 4;
        unsigned long merger_out_bytes = (unsigned long)groups * 896u * sizeof(float);
        mu_gpu_buf merger_out_buf = mu_gpu_scratch_alloc_a_ctx(ctx, merger_out_bytes);
        if (!merger_out_buf.ptr) { mu_gpu_cmd_discard(ctx); return -8; }

        if (profile_split) {
            mu_gpu_cmd_set_label(ctx, "vision_profile_merger_norm");
        }
        rc = mu_gpu_layernorm_bf16_rows_ctx(ctx, current_in, ln_w_buf, ln_b_buf, rows, 1280, 1e-6f, temp_normed);
        if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
        if (profile_split) {
            rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_merger_norm",
                                             "vision_profile_merger_merge4");
            if (rc != 0) return rc;
        }

        // Reuse temp_fc1 for merge4 output (groups * 5120 floats)
        rc = mu_gpu_vision_merge4_ctx(ctx, temp_normed, rows, temp_fc1);
        if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
        if (profile_split) {
            rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_merger_merge4",
                                             "vision_profile_merger_fc0_gelu");
            if (rc != 0) return rc;
        }

        // Dense layer: fc0 (groups * 5120 -> groups * 5120) with fused GELU
        // Reuse temp_fc1_act for activated
        rc = mu_gpu_dense_bf16_bias_rows_gelu_ctx(ctx, temp_fc1, fc0_w_buf, fc0_b_buf, groups, 5120, 5120, temp_fc1_act);
        if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }
        if (profile_split) {
            rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_merger_fc0_gelu",
                                             "vision_profile_merger_fc2");
            if (rc != 0) return rc;
        }

        // Dense layer: fc2 (groups * 5120 -> groups * 896)
        rc = mu_gpu_dense_bf16_bias_rows_ctx(ctx, temp_fc1_act, fc2_w_buf, fc2_b_buf, groups, 5120, 896, merger_out_buf);
        if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

        if (profile_split) {
            rc = mu_gpu_profile_commit_stage(gpu, &ctx, "vision_profile_merger_fc2", NULL);
        } else {
            rc = mu_gpu_cmd_commit_and_wait(ctx);
        }
        if (rc == 0) {
            mu_gpu_buf_copy_from(out, merger_out_buf, merger_out_bytes);
        }
        return rc;
    }
}

static int mu_gpu_text_layers_mlp_seq_engine(mu_gpu *gpu, void *engine,
                                            const float *hidden_states_cpu,
                                            int n_ids, const int *position_ids,
                                            int n_layers,
                                            int cache_cap,
                                            mu_gpu_kv_cache *gpu_cache,
                                            float *out,
                                            bool last_only) {
    if (!gpu || !engine || !hidden_states_cpu || n_ids <= 0 || n_layers <= 0) return -1;

    @autoreleasepool {
        mu_gpu_cmd_ctx *ctx = NULL;
        int rc = mu_gpu_cmd_begin(gpu, &ctx);
        if (rc != 0) return rc;
        mu_gpu_cmd_set_label(ctx, "text_prefill_seq");

        const int hidden = 896;
        const int inter = 4864;
        const float eps = 1e-6f;

        unsigned long embed_bytes = (unsigned long)n_ids * (unsigned long)hidden * sizeof(float);
        unsigned long kv_bytes = (unsigned long)n_ids * 128u * sizeof(float);
        unsigned long mlp_bytes = (unsigned long)n_ids * (unsigned long)inter * sizeof(float);

        mu_gpu_buf ping_buf = mu_gpu_scratch_alloc_a_ctx(ctx, embed_bytes);
        mu_gpu_buf pong_buf = mu_gpu_scratch_alloc_b_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_normed = mu_gpu_scratch_alloc_a_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_q = mu_gpu_scratch_alloc_a_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_k = mu_gpu_scratch_alloc_a_ctx(ctx, kv_bytes);
        mu_gpu_buf temp_v = mu_gpu_scratch_alloc_a_ctx(ctx, kv_bytes);
        mu_gpu_buf temp_attn = mu_gpu_scratch_alloc_b_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_proj = mu_gpu_scratch_alloc_b_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_res1 = mu_gpu_scratch_alloc_b_ctx(ctx, embed_bytes);
        mu_gpu_buf temp_fc1 = mu_gpu_scratch_alloc_a_ctx(ctx, mlp_bytes);
        mu_gpu_buf temp_fc1_act = mu_gpu_scratch_alloc_b_ctx(ctx, mlp_bytes);

        unsigned long pos_bytes = (unsigned long)n_ids * 3 * sizeof(int);
        mu_gpu_buf pos_buf = mu_gpu_scratch_alloc_a_ctx(ctx, pos_bytes);

        if (!ping_buf.ptr || !pong_buf.ptr || !temp_normed.ptr || !temp_q.ptr || !temp_k.ptr ||
            !temp_v.ptr || !temp_attn.ptr || !temp_proj.ptr || !temp_res1.ptr ||
            !temp_fc1.ptr || !temp_fc1_act.ptr || !pos_buf.ptr) {
            mu_gpu_cmd_discard(ctx);
            return -2;
        }

        mu_gpu_buf_copy_to(ping_buf, hidden_states_cpu, embed_bytes);

        // Prepare position IDs buffer on GPU
        int *temp_pos = (int *)malloc(pos_bytes);
        if (!temp_pos) { mu_gpu_cmd_discard(ctx); return -3; }
        for (int i = 0; i < n_ids; i++) {
            temp_pos[0 * n_ids + i] = position_ids ? position_ids[0 * n_ids + i] : i;
            temp_pos[1 * n_ids + i] = position_ids ? position_ids[1 * n_ids + i] : i;
            temp_pos[2 * n_ids + i] = position_ids ? position_ids[2 * n_ids + i] : i;
        }
        mu_gpu_buf_copy_to(pos_buf, temp_pos, pos_bytes);
        free(temp_pos);

        mu_gpu_buf current_in = ping_buf;
        mu_gpu_buf current_out = pong_buf;

        for (int layer = 0; layer < n_layers; layer++) {
            if (getenv("MU_METAL_DEBUG")) {
                fprintf(stderr, "mu metal stage: text_layer%d_seq_input_norm\n", layer);
                fprintf(stderr, "mu metal stage: text_layer%d_seq_qkv\n", layer);
                fprintf(stderr, "mu metal stage: text_layer%d_seq_attn\n", layer);
                fprintf(stderr, "mu metal stage: text_layer%d_seq_mlp_residual\n", layer);
            }
            const unsigned short *input_norm = mu_engine_get_text_layer_tensor(engine, layer, "input_layernorm.weight", 1, hidden, 0);
            const unsigned short *post_norm = mu_engine_get_text_layer_tensor(engine, layer, "post_attention_layernorm.weight", 1, hidden, 0);
            const unsigned short *qw = mu_engine_get_text_layer_tensor(engine, layer, "self_attn.q_proj.weight", 2, hidden, hidden);
            const unsigned short *qb = mu_engine_get_text_layer_tensor(engine, layer, "self_attn.q_proj.bias", 1, hidden, 0);
            const unsigned short *kw = mu_engine_get_text_layer_tensor(engine, layer, "self_attn.k_proj.weight", 2, 128, hidden);
            const unsigned short *kb = mu_engine_get_text_layer_tensor(engine, layer, "self_attn.k_proj.bias", 1, 128, 0);
            const unsigned short *vw = mu_engine_get_text_layer_tensor(engine, layer, "self_attn.v_proj.weight", 2, 128, hidden);
            const unsigned short *vb = mu_engine_get_text_layer_tensor(engine, layer, "self_attn.v_proj.bias", 1, 128, 0);
            const unsigned short *ow = mu_engine_get_text_layer_tensor(engine, layer, "self_attn.o_proj.weight", 2, hidden, hidden);
            const unsigned short *gate_w = mu_engine_get_text_layer_tensor(engine, layer, "mlp.gate_proj.weight", 2, inter, hidden);
            const unsigned short *up_w = mu_engine_get_text_layer_tensor(engine, layer, "mlp.up_proj.weight", 2, inter, hidden);
            const unsigned short *down_w = mu_engine_get_text_layer_tensor(engine, layer, "mlp.down_proj.weight", 2, hidden, inter);

            if (!input_norm || !post_norm || !qw || !qb || !kw || !kb || !vw || !vb ||
                !ow || !gate_w || !up_w || !down_w) {
                mu_gpu_cmd_discard(ctx);
                return -4;
            }

            mu_gpu_buf input_norm_buf = mu_gpu_get_weight_buf(gpu, input_norm, hidden * sizeof(unsigned short));
            mu_gpu_buf post_norm_buf = mu_gpu_get_weight_buf(gpu, post_norm, hidden * sizeof(unsigned short));
            mu_gpu_buf qw_buf = mu_gpu_get_weight_buf(gpu, qw, hidden * hidden * sizeof(unsigned short));
            mu_gpu_buf qb_buf = mu_gpu_get_weight_buf(gpu, qb, hidden * sizeof(unsigned short));
            mu_gpu_buf kw_buf = mu_gpu_get_weight_buf(gpu, kw, 128 * hidden * sizeof(unsigned short));
            mu_gpu_buf kb_buf = mu_gpu_get_weight_buf(gpu, kb, 128 * sizeof(unsigned short));
            mu_gpu_buf vw_buf = mu_gpu_get_weight_buf(gpu, vw, 128 * hidden * sizeof(unsigned short));
            mu_gpu_buf vb_buf = mu_gpu_get_weight_buf(gpu, vb, 128 * sizeof(unsigned short));
            mu_gpu_buf ow_buf = mu_gpu_get_weight_buf(gpu, ow, hidden * hidden * sizeof(unsigned short));
            mu_gpu_buf gate_w_buf = mu_gpu_get_weight_buf(gpu, gate_w, inter * hidden * sizeof(unsigned short));
            mu_gpu_buf up_w_buf = mu_gpu_get_weight_buf(gpu, up_w, inter * hidden * sizeof(unsigned short));
            mu_gpu_buf down_w_buf = mu_gpu_get_weight_buf(gpu, down_w, hidden * inter * sizeof(unsigned short));

            if (!input_norm_buf.ptr || !post_norm_buf.ptr || !qw_buf.ptr || !qb_buf.ptr ||
                !kw_buf.ptr || !kb_buf.ptr || !vw_buf.ptr || !vb_buf.ptr || !ow_buf.ptr ||
                !gate_w_buf.ptr || !up_w_buf.ptr || !down_w_buf.ptr) {
                mu_gpu_cmd_discard(ctx);
                return -5;
            }

            rc = mu_gpu_rmsnorm_bf16_rows_ctx(ctx, current_in, input_norm_buf, n_ids, hidden, eps, temp_normed);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_dense_f32_bias_rows_ctx(ctx, temp_normed, qw_buf, qb_buf, n_ids, hidden, hidden, temp_q);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_dense_f32_bias_rows_ctx(ctx, temp_normed, kw_buf, kb_buf, n_ids, hidden, 128, temp_k);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_dense_f32_bias_rows_ctx(ctx, temp_normed, vw_buf, vb_buf, n_ids, hidden, 128, temp_v);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            bool use_prefill_flash = gpu_cache && getenv("MU_TEXT_PREFILL_ATTN_NO_FLASH") == NULL;
            if (gpu_cache) {
                [ctx->encoder setComputePipelineState:gpu->text_prefill_rope_cache_update];
                [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)temp_k.ptr offset:temp_k.offset atIndex:0];
                [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)temp_v.ptr offset:temp_v.offset atIndex:1];
                [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)pos_buf.ptr offset:pos_buf.offset atIndex:2];
                [ctx->encoder setBuffer:gpu_cache->k_cache offset:0 atIndex:3];
                [ctx->encoder setBuffer:gpu_cache->v_cache offset:0 atIndex:4];
                [ctx->encoder setBytes:&n_ids length:sizeof(n_ids) atIndex:5];
                [ctx->encoder setBytes:&cache_cap length:sizeof(cache_cap) atIndex:6];
                [ctx->encoder setBytes:&layer length:sizeof(layer) atIndex:7];

                MTLSize grid = MTLSizeMake((NSUInteger)n_ids, 2, 1);
                MTLSize threads = MTLSizeMake(1, 2, 1);
                [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
            }

            if (use_prefill_flash) {
                if (position_ids) {
                    [ctx->encoder setComputePipelineState:gpu->text_prefill_attn_pos_flash];
                    [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)temp_q.ptr offset:temp_q.offset atIndex:0];
                    [ctx->encoder setBuffer:gpu_cache->k_cache offset:0 atIndex:1];
                    [ctx->encoder setBuffer:gpu_cache->v_cache offset:0 atIndex:2];
                    [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)pos_buf.ptr offset:pos_buf.offset atIndex:3];
                    [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)temp_attn.ptr offset:temp_attn.offset atIndex:4];
                    [ctx->encoder setBytes:&n_ids length:sizeof(n_ids) atIndex:5];
                    [ctx->encoder setBytes:&cache_cap length:sizeof(cache_cap) atIndex:6];
                    [ctx->encoder setBytes:&layer length:sizeof(layer) atIndex:7];
                } else {
                    [ctx->encoder setComputePipelineState:gpu->text_prefill_attn_flash];
                    [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)temp_q.ptr offset:temp_q.offset atIndex:0];
                    [ctx->encoder setBuffer:gpu_cache->k_cache offset:0 atIndex:1];
                    [ctx->encoder setBuffer:gpu_cache->v_cache offset:0 atIndex:2];
                    [ctx->encoder setBuffer:(__bridge id<MTLBuffer>)temp_attn.ptr offset:temp_attn.offset atIndex:3];
                    [ctx->encoder setBytes:&n_ids length:sizeof(n_ids) atIndex:4];
                    [ctx->encoder setBytes:&cache_cap length:sizeof(cache_cap) atIndex:5];
                    [ctx->encoder setBytes:&layer length:sizeof(layer) atIndex:6];
                }
                MTLSize grid = MTLSizeMake(((NSUInteger)n_ids + 31u) / 32u, 14, 1);
                MTLSize threads = MTLSizeMake(32, 1, 1);
                [ctx->encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
                rc = 0;
            } else {
                if (position_ids) {
                    rc = mu_gpu_text_attn_seq_pos_ctx(ctx, temp_q, temp_k, temp_v, pos_buf, n_ids, temp_attn);
                } else {
                    rc = mu_gpu_text_attn_seq_ctx(ctx, temp_q, temp_k, temp_v, n_ids, temp_attn);
                }
            }
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_dense_f32_rows_ctx(ctx, temp_attn, ow_buf, n_ids, hidden, hidden, temp_proj);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_add_f32_ctx(ctx, current_in, temp_proj, temp_res1, n_ids * hidden);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_rmsnorm_bf16_rows_ctx(ctx, temp_res1, post_norm_buf, n_ids, hidden, eps, temp_normed);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_dense_f32_rows_ctx(ctx, temp_normed, gate_w_buf, n_ids, hidden, inter, temp_fc1);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_dense_f32_rows_ctx(ctx, temp_normed, up_w_buf, n_ids, hidden, inter, temp_fc1_act);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_silu_mul_f32_ctx(ctx, temp_fc1, temp_fc1_act, temp_fc1, n_ids * inter);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_dense_f32_rows_ctx(ctx, temp_fc1, down_w_buf, n_ids, inter, hidden, temp_proj);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            rc = mu_gpu_add_f32_ctx(ctx, temp_res1, temp_proj, current_out, n_ids * hidden);
            if (rc != 0) { mu_gpu_cmd_discard(ctx); return rc; }

            // Swap
            mu_gpu_buf tmp = current_in;
            current_in = current_out;
            current_out = tmp;
        }

        rc = mu_gpu_cmd_commit_and_wait(ctx);
        if (rc == 0) {
            if (last_only) {
                mu_gpu_buf last_token_buf = current_in;
                last_token_buf.offset += (size_t)(n_ids - 1) * (size_t)hidden * sizeof(float);
                mu_gpu_buf_copy_from(out, last_token_buf, hidden * sizeof(float));
            } else {
                mu_gpu_buf_copy_from(out, current_in, embed_bytes);
            }
        }
        return rc;
    }
}

int mu_gpu_text_layers_mlp_seq_from_hidden(mu_gpu *gpu, void *engine,
                                           const float *initial_hidden,
                                           int n_ids, const int *position_ids,
                                           int n_layers, float *out) {
    return mu_gpu_text_layers_mlp_seq_engine(gpu, engine, initial_hidden,
                                             n_ids, position_ids, n_layers,
                                             0, NULL, out, false);
}

int mu_gpu_text_prefill_cache_from_embeddings(mu_gpu *gpu, void *engine,
                                              const float *hidden_states,
                                              int n_ids, const int *position_ids,
                                              int cache_cap,
                                              mu_gpu_kv_cache *gpu_cache,
                                              float *last_hidden_out) {
    return mu_gpu_text_layers_mlp_seq_engine(gpu, engine, hidden_states,
                                             n_ids, position_ids,
                                             mu_engine_text_layers((const mu_engine *)engine),
                                             cache_cap, gpu_cache, last_hidden_out, true);
}

int mu_gpu_text_layers_mlp_seq(mu_gpu *gpu, void *engine,
                               const int *input_ids, int n_ids,
                               int n_layers, float *out) {
    const int hidden = 896;
    const int vocab = 151936;
    const unsigned short *embed = mu_engine_get_vision_merger_tensor(engine, "model.embed_tokens.weight", 2, vocab, hidden);
    if (!embed) return -10;

    float *initial_hidden = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    if (!initial_hidden) return -11;

    for (int s = 0; s < n_ids; s++) {
        int id = input_ids[s];
        if (id < 0 || id >= vocab) {
            free(initial_hidden);
            return -12;
        }
        const unsigned short *row = embed + (size_t)id * hidden;
        for (int i = 0; i < hidden; i++) {
            initial_hidden[(size_t)s * hidden + i] = mu_gpu_bf16_to_f32(row[i]);
        }
    }

    int rc = mu_gpu_text_layers_mlp_seq_engine(gpu, engine, initial_hidden,
                                               n_ids, NULL, n_layers,
                                               0, NULL, out, false);
    free(initial_hidden);
    return rc;
}

// ==========================================
// C-compatible _ctx operator implementations
// ==========================================

int mu_gpu_rmsnorm_bf16_probe_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf weight,
                                  mu_gpu_buf out, int n, float eps) {
    if (!ctx || !x.ptr || !weight.ptr || !out.ptr || n <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weight.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->rmsnorm_bf16_probe];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:w_buf offset:weight.offset atIndex:1];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
    [ctx->encoder setBytes:&n length:sizeof(n) atIndex:3];
    [ctx->encoder setBytes:&eps length:sizeof(eps) atIndex:4];

    NSUInteger width = ctx->gpu->rmsnorm_bf16_probe.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)n) width = (NSUInteger)n;
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_rmsnorm_bf16_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf weight,
                                 int rows, int cols, float eps, mu_gpu_buf out) {
    if (!ctx || !x.ptr || !weight.ptr || !out.ptr || rows <= 0 || cols <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weight.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->rmsnorm_bf16_rows];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:w_buf offset:weight.offset atIndex:1];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
    [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    [ctx->encoder setBytes:&eps length:sizeof(eps) atIndex:4];

    NSUInteger width = ctx->gpu->rmsnorm_bf16_rows.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)cols) width = (NSUInteger)cols;
    MTLSize grid = MTLSizeMake((NSUInteger)cols, (NSUInteger)rows, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_dense_f32_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                              int x_rows, int cols, int out_cols, mu_gpu_buf out) {
    if (!ctx || !x.ptr || !w.ptr || !out.ptr || x_rows <= 0 || cols <= 0 || out_cols <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    bool request_f32_mps = getenv("MU_DENSE_F32_ROWS_MPS") != NULL;
    bool disable_f32_mps = getenv("MU_DENSE_F32_ROWS_NO_MPS") != NULL;
    if ((request_f32_mps || !disable_f32_mps) &&
        mu_gpu_dense_mps_text_shape(cols, out_cols)) {
        MPSMatrixMultiplication *kernel = mu_gpu_dense_mps_kernel(ctx->gpu, x_rows, cols, out_cols);
        if (kernel) {
            NSUInteger w_bytes = (NSUInteger)out_cols * (NSUInteger)cols * sizeof(unsigned short);
            id<MTLBuffer> w_f32_buf = mu_gpu_dense_mps_f32_weight(ctx->gpu, w_buf, w.offset, w_bytes);
            if (!w_f32_buf) return -3;

            @autoreleasepool {
                MPSMatrixDescriptor *x_desc =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)x_rows
                                                          columns:(NSUInteger)cols
                                                         rowBytes:(NSUInteger)cols * sizeof(float)
                                                         dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor *w_desc =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)out_cols
                                                          columns:(NSUInteger)cols
                                                         rowBytes:(NSUInteger)cols * sizeof(float)
                                                         dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor *out_desc =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)x_rows
                                                          columns:(NSUInteger)out_cols
                                                         rowBytes:(NSUInteger)out_cols * sizeof(float)
                                                         dataType:MPSDataTypeFloat32];
                MPSMatrix *x_matrix = [[MPSMatrix alloc] initWithBuffer:x_buf offset:x.offset descriptor:x_desc];
                MPSMatrix *w_matrix = [[MPSMatrix alloc] initWithBuffer:w_f32_buf descriptor:w_desc];
                MPSMatrix *out_matrix = [[MPSMatrix alloc] initWithBuffer:out_buf offset:out.offset descriptor:out_desc];
                if (!x_matrix || !w_matrix || !out_matrix) return -4;

                int rc = mu_gpu_cmd_end_encoder(ctx);
                if (rc != 0) return rc;
                [kernel encodeToCommandBuffer:ctx->command_buffer
                                    leftMatrix:x_matrix
                                   rightMatrix:w_matrix
                                  resultMatrix:out_matrix];
                rc = mu_gpu_cmd_begin_encoder(ctx);
                if (rc != 0) return rc;
                return 0;
            }
        }
    }

    [ctx->encoder setComputePipelineState:ctx->gpu->dense_f32_rows];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
    [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    [ctx->encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:4];

    NSUInteger width = ctx->gpu->dense_f32_rows.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)out_cols) width = (NSUInteger)out_cols;
    MTLSize grid = MTLSizeMake((NSUInteger)out_cols, (NSUInteger)x_rows, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_dense_f32_bias_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                   mu_gpu_buf bias, int x_rows, int cols, int out_cols, mu_gpu_buf out) {
    if (!ctx || !x.ptr || !w.ptr || !bias.ptr || !out.ptr || x_rows <= 0 || cols <= 0 || out_cols <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w.ptr;
    id<MTLBuffer> bias_buf = (__bridge id<MTLBuffer>)bias.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->dense_f32_bias_rows];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
    [ctx->encoder setBuffer:bias_buf offset:bias.offset atIndex:2];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
    [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    [ctx->encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:5];

    NSUInteger width = ctx->gpu->dense_f32_bias_rows.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)out_cols) width = (NSUInteger)out_cols;
    MTLSize grid = MTLSizeMake((NSUInteger)out_cols, (NSUInteger)x_rows, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_text_attn_seq_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q, mu_gpu_buf k,
                             mu_gpu_buf v, int seq, mu_gpu_buf out) {
    if (!ctx || !q.ptr || !k.ptr || !v.ptr || !out.ptr || seq <= 0) return -1;
    id<MTLBuffer> q_buf = (__bridge id<MTLBuffer>)q.ptr;
    id<MTLBuffer> k_buf = (__bridge id<MTLBuffer>)k.ptr;
    id<MTLBuffer> v_buf = (__bridge id<MTLBuffer>)v.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->text_attn_seq];
    [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
    [ctx->encoder setBuffer:k_buf offset:k.offset atIndex:1];
    [ctx->encoder setBuffer:v_buf offset:v.offset atIndex:2];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
    [ctx->encoder setBytes:&seq length:sizeof(seq) atIndex:4];

    NSUInteger total = (NSUInteger)seq * 896u;
    NSUInteger width = ctx->gpu->text_attn_seq.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > total) width = total;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_text_attn_seq_pos_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q, mu_gpu_buf k,
                                 mu_gpu_buf v, mu_gpu_buf position_ids, int seq, mu_gpu_buf out) {
    if (!ctx || !q.ptr || !k.ptr || !v.ptr || !position_ids.ptr || !out.ptr || seq <= 0) return -1;
    id<MTLBuffer> q_buf = (__bridge id<MTLBuffer>)q.ptr;
    id<MTLBuffer> k_buf = (__bridge id<MTLBuffer>)k.ptr;
    id<MTLBuffer> v_buf = (__bridge id<MTLBuffer>)v.ptr;
    id<MTLBuffer> pos_buf = (__bridge id<MTLBuffer>)position_ids.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->text_attn_seq_pos];
    [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
    [ctx->encoder setBuffer:k_buf offset:k.offset atIndex:1];
    [ctx->encoder setBuffer:v_buf offset:v.offset atIndex:2];
    [ctx->encoder setBuffer:pos_buf offset:position_ids.offset atIndex:3];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:4];
    [ctx->encoder setBytes:&seq length:sizeof(seq) atIndex:5];

    NSUInteger width = ctx->gpu->text_attn_seq_pos.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)seq) width = (NSUInteger)seq;
    MTLSize grid = MTLSizeMake((NSUInteger)seq, 14, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_dense_f32_bias_probe_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                    mu_gpu_buf bias, mu_gpu_buf out, int rows, int cols) {
    if (!ctx || !x.ptr || !w.ptr || !bias.ptr || !out.ptr || rows <= 0 || cols <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w.ptr;
    id<MTLBuffer> bias_buf = (__bridge id<MTLBuffer>)bias.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    bool use_simd = getenv("MU_USE_SIMD") != NULL;
    if (use_simd && ctx->gpu->dense_f32_bias_probe_simd) {
        [ctx->encoder setComputePipelineState:ctx->gpu->dense_f32_bias_probe_simd];
        [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
        [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
        [ctx->encoder setBuffer:bias_buf offset:bias.offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:4];

        MTLSize grid = MTLSizeMake(32, (NSUInteger)rows, 1);
        MTLSize threads = MTLSizeMake(32, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    } else {
        [ctx->encoder setComputePipelineState:ctx->gpu->dense_f32_bias_probe];
        [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
        [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
        [ctx->encoder setBuffer:bias_buf offset:bias.offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:4];

        NSUInteger width = ctx->gpu->dense_f32_bias_probe.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)rows) width = (NSUInteger)rows;
        MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }
    return 0;
}

int mu_gpu_text_decode_qkv_proj_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x,
                                    mu_gpu_buf qw, mu_gpu_buf qb,
                                    mu_gpu_buf kw, mu_gpu_buf kb,
                                    mu_gpu_buf vw, mu_gpu_buf vb,
                                    mu_gpu_buf q_out, mu_gpu_buf k_out, mu_gpu_buf v_out,
                                    int cols) {
    if (!ctx || !x.ptr || !qw.ptr || !qb.ptr || !kw.ptr || !kb.ptr || !vw.ptr || !vb.ptr ||
        !q_out.ptr || !k_out.ptr || !v_out.ptr || cols <= 0) return -1;
    if (!ctx->gpu->text_decode_qkv_proj_simd) return -2;

    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> qw_buf = (__bridge id<MTLBuffer>)qw.ptr;
    id<MTLBuffer> qb_buf = (__bridge id<MTLBuffer>)qb.ptr;
    id<MTLBuffer> kw_buf = (__bridge id<MTLBuffer>)kw.ptr;
    id<MTLBuffer> kb_buf = (__bridge id<MTLBuffer>)kb.ptr;
    id<MTLBuffer> vw_buf = (__bridge id<MTLBuffer>)vw.ptr;
    id<MTLBuffer> vb_buf = (__bridge id<MTLBuffer>)vb.ptr;
    id<MTLBuffer> q_out_buf = (__bridge id<MTLBuffer>)q_out.ptr;
    id<MTLBuffer> k_out_buf = (__bridge id<MTLBuffer>)k_out.ptr;
    id<MTLBuffer> v_out_buf = (__bridge id<MTLBuffer>)v_out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->text_decode_qkv_proj_simd];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:qw_buf offset:qw.offset atIndex:1];
    [ctx->encoder setBuffer:qb_buf offset:qb.offset atIndex:2];
    [ctx->encoder setBuffer:kw_buf offset:kw.offset atIndex:3];
    [ctx->encoder setBuffer:kb_buf offset:kb.offset atIndex:4];
    [ctx->encoder setBuffer:vw_buf offset:vw.offset atIndex:5];
    [ctx->encoder setBuffer:vb_buf offset:vb.offset atIndex:6];
    [ctx->encoder setBuffer:q_out_buf offset:q_out.offset atIndex:7];
    [ctx->encoder setBuffer:k_out_buf offset:k_out.offset atIndex:8];
    [ctx->encoder setBuffer:v_out_buf offset:v_out.offset atIndex:9];
    [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:10];

    MTLSize grid = MTLSizeMake(32, 1152, 1);
    MTLSize threads = MTLSizeMake(32, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];

    return 0;
}

int mu_gpu_text_decode_qkv_rope_cache_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x,
                                          mu_gpu_buf qw, mu_gpu_buf qb,
                                          mu_gpu_buf kw, mu_gpu_buf kb,
                                          mu_gpu_buf vw, mu_gpu_buf vb,
                                          mu_gpu_buf q_out,
                                          mu_gpu_kv_cache *cache, int layer,
                                          int cache_pos, const int pos3[3],
                                          int cols) {
    if (!ctx || !x.ptr || !qw.ptr || !qb.ptr || !kw.ptr || !kb.ptr || !vw.ptr || !vb.ptr ||
        !q_out.ptr || !cache || !pos3 || cols <= 0 ||
        layer < 0 || layer >= cache->layers ||
        cache_pos < 0 || cache_pos >= cache->cap) return -1;
    if (!ctx->gpu->text_decode_qkv_rope_cache_simd) return -2;

    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> qw_buf = (__bridge id<MTLBuffer>)qw.ptr;
    id<MTLBuffer> qb_buf = (__bridge id<MTLBuffer>)qb.ptr;
    id<MTLBuffer> kw_buf = (__bridge id<MTLBuffer>)kw.ptr;
    id<MTLBuffer> kb_buf = (__bridge id<MTLBuffer>)kb.ptr;
    id<MTLBuffer> vw_buf = (__bridge id<MTLBuffer>)vw.ptr;
    id<MTLBuffer> vb_buf = (__bridge id<MTLBuffer>)vb.ptr;
    id<MTLBuffer> q_out_buf = (__bridge id<MTLBuffer>)q_out.ptr;
    NSUInteger kv_offset = (NSUInteger)layer * (NSUInteger)cache->cap * 128u * sizeof(float);

    [ctx->encoder setComputePipelineState:ctx->gpu->text_decode_qkv_rope_cache_simd];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:qw_buf offset:qw.offset atIndex:1];
    [ctx->encoder setBuffer:qb_buf offset:qb.offset atIndex:2];
    [ctx->encoder setBuffer:kw_buf offset:kw.offset atIndex:3];
    [ctx->encoder setBuffer:kb_buf offset:kb.offset atIndex:4];
    [ctx->encoder setBuffer:vw_buf offset:vw.offset atIndex:5];
    [ctx->encoder setBuffer:vb_buf offset:vb.offset atIndex:6];
    [ctx->encoder setBuffer:q_out_buf offset:q_out.offset atIndex:7];
    [ctx->encoder setBuffer:cache->k_cache offset:kv_offset atIndex:8];
    [ctx->encoder setBuffer:cache->v_cache offset:kv_offset atIndex:9];
    [ctx->encoder setBytes:pos3 length:3 * sizeof(pos3[0]) atIndex:10];
    [ctx->encoder setBytes:&cache_pos length:sizeof(cache_pos) atIndex:11];
    [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:12];

    MTLSize grid = MTLSizeMake(32, 640, 1);
    MTLSize threads = MTLSizeMake(32, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_dense_probe_add_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                               mu_gpu_buf residual, mu_gpu_buf out, int rows, int cols) {
    if (!ctx || !x.ptr || !w.ptr || !residual.ptr || !out.ptr || rows <= 0 || cols <= 0) return -1;
    if (getenv("MU_TEXT_DECODE_NO_PROBE_ADD_FUSION") || !ctx->gpu->dense_probe_add_simd) {
        mu_gpu_buf proj = mu_gpu_scratch_alloc_a_ctx(ctx, (unsigned long)rows * sizeof(float));
        if (!proj.ptr) return -2;
        int rc = mu_gpu_dense_probe_ctx(ctx, x, w, proj, rows, cols);
        if (rc == 0) rc = mu_gpu_add_f32_ctx(ctx, residual, proj, out, rows);
        return rc;
    }

    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w.ptr;
    id<MTLBuffer> res_buf = (__bridge id<MTLBuffer>)residual.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->dense_probe_add_simd];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
    [ctx->encoder setBuffer:res_buf offset:residual.offset atIndex:2];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
    [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:4];

    MTLSize grid = MTLSizeMake(32, (NSUInteger)rows, 1);
    MTLSize threads = MTLSizeMake(32, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];

    return 0;
}

int mu_gpu_dense_probe_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                           mu_gpu_buf out, int rows, int cols) {
    if (!ctx || !x.ptr || !w.ptr || !out.ptr || rows <= 0 || cols <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    bool use_simd = getenv("MU_USE_SIMD") != NULL;
    if (use_simd && ctx->gpu->dense_probe_simd) {
        [ctx->encoder setComputePipelineState:ctx->gpu->dense_probe_simd];
        [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
        [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
        [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:3];

        MTLSize grid = MTLSizeMake(32, (NSUInteger)rows, 1);
        MTLSize threads = MTLSizeMake(32, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    } else {
        [ctx->encoder setComputePipelineState:ctx->gpu->dense_probe];
        [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
        [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
        [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:3];

        NSUInteger width = ctx->gpu->dense_probe.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)rows) width = (NSUInteger)rows;
        MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }
    return 0;
}

int mu_gpu_add_f32_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf a, mu_gpu_buf b,
                       mu_gpu_buf out, int n) {
    if (!ctx || !a.ptr || !b.ptr || !out.ptr || n <= 0) return -1;
    id<MTLBuffer> a_buf = (__bridge id<MTLBuffer>)a.ptr;
    id<MTLBuffer> b_buf = (__bridge id<MTLBuffer>)b.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->add_f32];
    [ctx->encoder setBuffer:a_buf offset:a.offset atIndex:0];
    [ctx->encoder setBuffer:b_buf offset:b.offset atIndex:1];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
    [ctx->encoder setBytes:&n length:sizeof(n) atIndex:3];

    NSUInteger width = ctx->gpu->add_f32.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)n) width = (NSUInteger)n;
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_silu_mul_f32_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf gate, mu_gpu_buf up,
                            mu_gpu_buf out, int n) {
    if (!ctx || !gate.ptr || !up.ptr || !out.ptr || n <= 0) return -1;
    id<MTLBuffer> gate_buf = (__bridge id<MTLBuffer>)gate.ptr;
    id<MTLBuffer> up_buf = (__bridge id<MTLBuffer>)up.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->silu_mul_f32];
    [ctx->encoder setBuffer:gate_buf offset:gate.offset atIndex:0];
    [ctx->encoder setBuffer:up_buf offset:up.offset atIndex:1];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
    [ctx->encoder setBytes:&n length:sizeof(n) atIndex:3];

    NSUInteger width = ctx->gpu->silu_mul_f32.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)n) width = (NSUInteger)n;
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_text_attn_cached_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q,
                                const float *k_cache, const float *v_cache,
                                int cache_len, mu_gpu_buf out) {
    if (!ctx || !q.ptr || !k_cache || !v_cache || !out.ptr || cache_len <= 0) return -1;
    id<MTLBuffer> q_buf = (__bridge id<MTLBuffer>)q.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    NSUInteger kv_bytes = (NSUInteger)cache_len * 128u * sizeof(float);
    NSUInteger k_offset = 0, v_offset = 0;
    id<MTLBuffer> k_buf = mu_scratch_alloc_a(&ctx->alloc, kv_bytes, &k_offset);
    if (k_buf) {
        memcpy((char *)[k_buf contents] + k_offset, k_cache, kv_bytes);
    } else {
        k_buf = [ctx->gpu->device newBufferWithBytes:k_cache length:kv_bytes options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> v_buf = mu_scratch_alloc_a(&ctx->alloc, kv_bytes, &v_offset);
    if (v_buf) {
        memcpy((char *)[v_buf contents] + v_offset, v_cache, kv_bytes);
    } else {
        v_buf = [ctx->gpu->device newBufferWithBytes:v_cache length:kv_bytes options:MTLResourceStorageModeShared];
    }

    bool disable_simd = getenv("MU_TEXT_ATTN_CACHED_NO_SIMD") != NULL;
    bool use_simd = !disable_simd;
    if (use_simd && ctx->gpu->text_attn_cached_simd) {
        [ctx->encoder setComputePipelineState:ctx->gpu->text_attn_cached_simd];
        [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
        [ctx->encoder setBuffer:k_buf offset:k_offset atIndex:1];
        [ctx->encoder setBuffer:v_buf offset:v_offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&cache_len length:sizeof(cache_len) atIndex:4];

        MTLSize grid = MTLSizeMake(14 * 32, 1, 1);
        MTLSize threads = MTLSizeMake(32, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    } else {
        [ctx->encoder setComputePipelineState:ctx->gpu->text_attn_cached];
        [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
        [ctx->encoder setBuffer:k_buf offset:k_offset atIndex:1];
        [ctx->encoder setBuffer:v_buf offset:v_offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&cache_len length:sizeof(cache_len) atIndex:4];

        MTLSize grid = MTLSizeMake(14, 1, 1);
        MTLSize threads = MTLSizeMake(1, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }
    return 0;
}

int mu_gpu_layernorm_bf16_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf weight,
                                   mu_gpu_buf bias, int rows, int cols, float eps, mu_gpu_buf out) {
    if (!ctx || !x.ptr || !weight.ptr || !bias.ptr || !out.ptr || rows <= 0 || cols <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)weight.ptr;
    id<MTLBuffer> bias_buf = (__bridge id<MTLBuffer>)bias.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->layernorm_bf16_rows];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:w_buf offset:weight.offset atIndex:1];
    [ctx->encoder setBuffer:bias_buf offset:bias.offset atIndex:2];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
    [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    [ctx->encoder setBytes:&eps length:sizeof(eps) atIndex:5];

    NSUInteger width = ctx->gpu->layernorm_bf16_rows.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)cols) width = (NSUInteger)cols;
    MTLSize grid = MTLSizeMake((NSUInteger)cols, (NSUInteger)rows, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

static bool mu_gpu_dense_mps_shape(int cols, int out_cols) {
    return (cols == 1280 && (out_cols == 1280 || out_cols == 2560 || out_cols == 3840 || out_cols == 5120)) ||
           (cols == 5120 && out_cols == 1280);
}

static bool mu_gpu_dense_mps_text_shape(int cols, int out_cols) {
    return (cols == 896 && (out_cols == 128 || out_cols == 896 || out_cols == 4864)) ||
           (cols == 4864 && out_cols == 896);
}

static MPSMatrixMultiplication *mu_gpu_dense_mps_kernel(mu_gpu *gpu,
                                                       int rows, int cols, int out_cols) {
    if (!gpu || rows <= 0 || !(mu_gpu_dense_mps_shape(cols, out_cols) || mu_gpu_dense_mps_text_shape(cols, out_cols))) return nil;

    if (cols == 1280 && out_cols == 1280) {
        if (!gpu->dense_mps_1280_1280 || gpu->dense_mps_1280_1280_rows != rows) {
            gpu->dense_mps_1280_1280 =
                [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:(NSUInteger)rows
                                                  resultColumns:1280
                                                interiorColumns:1280
                                                          alpha:1.0
                                                           beta:0.0];
            gpu->dense_mps_1280_1280_rows = rows;
        }
        return gpu->dense_mps_1280_1280;
    }
    if (cols == 1280 && out_cols == 2560) {
        if (!gpu->dense_mps_1280_2560 || gpu->dense_mps_1280_2560_rows != rows) {
            gpu->dense_mps_1280_2560 =
                [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:(NSUInteger)rows
                                                  resultColumns:2560
                                                interiorColumns:1280
                                                          alpha:1.0
                                                           beta:0.0];
            gpu->dense_mps_1280_2560_rows = rows;
        }
        return gpu->dense_mps_1280_2560;
    }
    if (cols == 1280 && out_cols == 3840) {
        if (!gpu->dense_mps_1280_3840 || gpu->dense_mps_1280_3840_rows != rows) {
            gpu->dense_mps_1280_3840 =
                [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:(NSUInteger)rows
                                                  resultColumns:3840
                                                interiorColumns:1280
                                                          alpha:1.0
                                                           beta:0.0];
            gpu->dense_mps_1280_3840_rows = rows;
        }
        return gpu->dense_mps_1280_3840;
    }
    if (cols == 1280 && out_cols == 5120) {
        if (!gpu->dense_mps_1280_5120 || gpu->dense_mps_1280_5120_rows != rows) {
            gpu->dense_mps_1280_5120 =
                [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:(NSUInteger)rows
                                                  resultColumns:5120
                                                interiorColumns:1280
                                                          alpha:1.0
                                                           beta:0.0];
            gpu->dense_mps_1280_5120_rows = rows;
        }
        return gpu->dense_mps_1280_5120;
    }
    if (cols == 5120 && out_cols == 1280) {
        if (!gpu->dense_mps_5120_1280 || gpu->dense_mps_5120_1280_rows != rows) {
            gpu->dense_mps_5120_1280 =
                [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:(NSUInteger)rows
                                                  resultColumns:1280
                                                interiorColumns:5120
                                                          alpha:1.0
                                                           beta:0.0];
            gpu->dense_mps_5120_1280_rows = rows;
        }
        return gpu->dense_mps_5120_1280;
    }
    if (cols == 896 && out_cols == 896) {
        if (!gpu->dense_mps_896_896 || gpu->dense_mps_896_896_rows != rows) {
            gpu->dense_mps_896_896 =
                [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:(NSUInteger)rows
                                                  resultColumns:896
                                                interiorColumns:896
                                                          alpha:1.0
                                                           beta:0.0];
            gpu->dense_mps_896_896_rows = rows;
        }
        return gpu->dense_mps_896_896;
    }
    if (cols == 896 && out_cols == 128) {
        if (!gpu->dense_mps_896_128 || gpu->dense_mps_896_128_rows != rows) {
            gpu->dense_mps_896_128 =
                [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:(NSUInteger)rows
                                                  resultColumns:128
                                                interiorColumns:896
                                                          alpha:1.0
                                                           beta:0.0];
            gpu->dense_mps_896_128_rows = rows;
        }
        return gpu->dense_mps_896_128;
    }
    if (cols == 896 && out_cols == 4864) {
        if (!gpu->dense_mps_896_4864 || gpu->dense_mps_896_4864_rows != rows) {
            gpu->dense_mps_896_4864 =
                [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:(NSUInteger)rows
                                                  resultColumns:4864
                                                interiorColumns:896
                                                          alpha:1.0
                                                           beta:0.0];
            gpu->dense_mps_896_4864_rows = rows;
        }
        return gpu->dense_mps_896_4864;
    }
    if (cols == 4864 && out_cols == 896) {
        if (!gpu->dense_mps_4864_896 || gpu->dense_mps_4864_896_rows != rows) {
            gpu->dense_mps_4864_896 =
                [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                                  transposeLeft:NO
                                                 transposeRight:YES
                                                     resultRows:(NSUInteger)rows
                                                  resultColumns:896
                                                interiorColumns:4864
                                                          alpha:1.0
                                                           beta:0.0];
            gpu->dense_mps_4864_896_rows = rows;
        }
        return gpu->dense_mps_4864_896;
    }
    return nil;
}

static id<MTLBuffer> mu_gpu_dense_mps_f32_weight(mu_gpu *gpu, id<MTLBuffer> src,
                                                NSUInteger offset, NSUInteger length) {
    if (!gpu || !src || length == 0) return nil;
    for (int i = 0; i < gpu->dense_mps_weight_cache_count; i++) {
        mu_gpu_dense_mps_weight *entry = &gpu->dense_mps_weight_cache[i];
        if (entry->src == src && entry->offset == offset && entry->length == length) {
            return entry->f32;
        }
    }

    NSUInteger count = length / sizeof(unsigned short);
    NSUInteger f32_bytes = count * sizeof(float);
    id<MTLBuffer> f32 = [gpu->device newBufferWithLength:f32_bytes
                                                  options:MTLResourceStorageModeShared];
    if (!f32) return nil;
    const unsigned short *s = (const unsigned short *)((const char *)[src contents] + offset);
    float *d = (float *)[f32 contents];
    for (NSUInteger i = 0; i < count; i++) d[i] = mu_gpu_bf16_to_f32(s[i]);

    if (gpu->dense_mps_weight_cache_count < MU_GPU_DENSE_MPS_WEIGHT_CACHE_CAP) {
        mu_gpu_dense_mps_weight *entry =
            &gpu->dense_mps_weight_cache[gpu->dense_mps_weight_cache_count++];
        entry->src = src;
        entry->offset = offset;
        entry->length = length;
        entry->f32 = f32;
    }
    return f32;
}

static int mu_gpu_dense_f32_rows_mps(mu_gpu *gpu,
                                     id<MTLBuffer> x_buf, NSUInteger x_offset,
                                     id<MTLBuffer> w_buf, NSUInteger w_offset,
                                     id<MTLBuffer> out_buf, NSUInteger out_offset,
                                     int x_rows, int cols, int out_cols,
                                     float *out) {
    if (!gpu || !x_buf || !w_buf || !out_buf || !out ||
        x_rows <= 0 || cols <= 0 || out_cols <= 0 ||
        !mu_gpu_dense_mps_text_shape(cols, out_cols)) {
        return -1;
    }

    NSUInteger w_bytes = (NSUInteger)out_cols * (NSUInteger)cols * sizeof(unsigned short);
    id<MTLBuffer> w_f32_buf = mu_gpu_dense_mps_f32_weight(gpu, w_buf, w_offset, w_bytes);
    if (!w_f32_buf) return -2;

    @autoreleasepool {
        MPSMatrixMultiplication *kernel =
            [[MPSMatrixMultiplication alloc] initWithDevice:gpu->device
                                              transposeLeft:NO
                                             transposeRight:YES
                                                 resultRows:(NSUInteger)x_rows
                                              resultColumns:(NSUInteger)out_cols
                                            interiorColumns:(NSUInteger)cols
                                                      alpha:1.0
                                                       beta:0.0];
        if (!kernel) return -3;

        MPSMatrixDescriptor *x_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)x_rows
                                                  columns:(NSUInteger)cols
                                                 rowBytes:(NSUInteger)cols * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *w_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)out_cols
                                                  columns:(NSUInteger)cols
                                                 rowBytes:(NSUInteger)cols * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *out_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)x_rows
                                                  columns:(NSUInteger)out_cols
                                                 rowBytes:(NSUInteger)out_cols * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrix *x_matrix = [[MPSMatrix alloc] initWithBuffer:x_buf offset:x_offset descriptor:x_desc];
        MPSMatrix *w_matrix = [[MPSMatrix alloc] initWithBuffer:w_f32_buf descriptor:w_desc];
        MPSMatrix *out_matrix = [[MPSMatrix alloc] initWithBuffer:out_buf offset:out_offset descriptor:out_desc];
        if (!x_matrix || !w_matrix || !out_matrix) return -4;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -5;
        [kernel encodeToCommandBuffer:command_buffer
                            leftMatrix:x_matrix
                           rightMatrix:w_matrix
                          resultMatrix:out_matrix];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, (char *)[out_buf contents] + out_offset,
               (size_t)x_rows * (size_t)out_cols * sizeof(out[0]));
    }
    return 0;
}

static int mu_gpu_dense_bf16_bias_rows_mps_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x,
                                               mu_gpu_buf w, mu_gpu_buf bias,
                                               int x_rows, int cols, int out_cols,
                                               mu_gpu_buf out) {
    if (!ctx || !ctx->gpu || !ctx->gpu->dense_mps_bias_round) return -1;
    MPSMatrixMultiplication *kernel =
        mu_gpu_dense_mps_kernel(ctx->gpu, x_rows, cols, out_cols);
    if (!kernel) return -2;

    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w.ptr;
    id<MTLBuffer> bias_buf = (__bridge id<MTLBuffer>)bias.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;
    NSUInteger w_bytes = (NSUInteger)out_cols * (NSUInteger)cols * sizeof(unsigned short);
    id<MTLBuffer> w_f32_buf = mu_gpu_dense_mps_f32_weight(ctx->gpu, w_buf, w.offset, w_bytes);
    if (!w_f32_buf) return -3;

    @autoreleasepool {
        MPSMatrixDescriptor *x_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)x_rows
                                                  columns:(NSUInteger)cols
                                                 rowBytes:(NSUInteger)cols * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *w_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)out_cols
                                                  columns:(NSUInteger)cols
                                                 rowBytes:(NSUInteger)cols * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *out_desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)x_rows
                                                  columns:(NSUInteger)out_cols
                                                 rowBytes:(NSUInteger)out_cols * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrix *x_matrix = [[MPSMatrix alloc] initWithBuffer:x_buf offset:x.offset descriptor:x_desc];
        MPSMatrix *w_matrix = [[MPSMatrix alloc] initWithBuffer:w_f32_buf descriptor:w_desc];
        MPSMatrix *out_matrix = [[MPSMatrix alloc] initWithBuffer:out_buf offset:out.offset descriptor:out_desc];
        if (!x_matrix || !w_matrix || !out_matrix) return -4;

        int rc = mu_gpu_cmd_end_encoder(ctx);
        if (rc != 0) return rc;
        [kernel encodeToCommandBuffer:ctx->command_buffer
                            leftMatrix:x_matrix
                           rightMatrix:w_matrix
                          resultMatrix:out_matrix];
        rc = mu_gpu_cmd_begin_encoder(ctx);
        if (rc != 0) return rc;

        [ctx->encoder setComputePipelineState:ctx->gpu->dense_mps_bias_round];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:0];
        [ctx->encoder setBuffer:bias_buf offset:bias.offset atIndex:1];
        [ctx->encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:2];
        NSUInteger width = ctx->gpu->dense_mps_bias_round.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)out_cols) width = (NSUInteger)out_cols;
        MTLSize grid = MTLSizeMake((NSUInteger)out_cols, (NSUInteger)x_rows, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        return 0;
    }
}

int mu_gpu_dense_bf16_bias_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                    mu_gpu_buf bias, int x_rows, int cols, int out_cols, mu_gpu_buf out) {
    if (!ctx || !x.ptr || !w.ptr || !bias.ptr || !out.ptr || x_rows <= 0 || cols <= 0 || out_cols <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w.ptr;
    id<MTLBuffer> bias_buf = (__bridge id<MTLBuffer>)bias.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    bool request_mps = getenv("MU_DENSE_ROWS_MPS") != NULL;
    bool use_simdgroup = mu_gpu_dense_mps_shape(cols, out_cols) &&
                         ctx->gpu->dense_bf16_bias_rows_simdgroup &&
                         getenv("MU_DENSE_ROWS_NO_SIMDGROUP") == NULL;
    if (use_simdgroup && !request_mps) {
        [ctx->encoder setComputePipelineState:ctx->gpu->dense_bf16_bias_rows_simdgroup];
        [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
        [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
        [ctx->encoder setBuffer:bias_buf offset:bias.offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [ctx->encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:5];
        [ctx->encoder setBytes:&x_rows length:sizeof(x_rows) atIndex:6];

        MTLSize grid = MTLSizeMake(((NSUInteger)out_cols + 31u) / 32u, ((NSUInteger)x_rows + 7u) / 8u, 1);
        MTLSize threads = MTLSizeMake(32, 1, 1);
        [ctx->encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        return 0;
    }

    bool can_use_tiled = mu_gpu_dense_mps_shape(cols, out_cols) || mu_gpu_dense_mps_text_shape(cols, out_cols);
    bool disable_mps = getenv("MU_DENSE_ROWS_NO_MPS") != NULL;
    if ((request_mps || !disable_mps) && can_use_tiled) {
        return mu_gpu_dense_bf16_bias_rows_mps_ctx(ctx, x, w, bias,
                                                   x_rows, cols, out_cols, out);
    }
    bool use_rows_tiled = getenv("MU_DENSE_ROWS_TILED") != NULL &&
                          can_use_tiled &&
                          ctx->gpu->dense_bf16_bias_rows_tiled;
    bool use_rows_simd = !use_rows_tiled &&
                         getenv("MU_DENSE_ROWS_SIMD") != NULL &&
                         ctx->gpu->dense_bf16_bias_rows_simd;
    id<MTLComputePipelineState> pipeline =
        use_rows_tiled ? ctx->gpu->dense_bf16_bias_rows_tiled
                      : use_rows_simd ? ctx->gpu->dense_bf16_bias_rows_simd
                      : ctx->gpu->dense_bf16_bias_rows;

    [ctx->encoder setComputePipelineState:pipeline];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
    [ctx->encoder setBuffer:bias_buf offset:bias.offset atIndex:2];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
    [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    [ctx->encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:5];

    if (use_rows_tiled) {
        MTLSize grid = MTLSizeMake(((NSUInteger)out_cols + 7u) / 8u, (NSUInteger)x_rows, 1);
        MTLSize threads = MTLSizeMake(256, 1, 1);
        [ctx->encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        return 0;
    } else if (use_rows_simd) {
        MTLSize grid = MTLSizeMake((NSUInteger)out_cols * 32u, (NSUInteger)x_rows, 1);
        MTLSize threads = MTLSizeMake(32, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        return 0;
    }

    NSUInteger width = pipeline.threadExecutionWidth;
    if (width < 1) width = 1;
    MTLSize grid = MTLSizeMake((NSUInteger)out_cols, (NSUInteger)x_rows, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_dense_bf16_bias_rows_quick_gelu_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                               mu_gpu_buf bias, int x_rows, int cols, int out_cols, mu_gpu_buf out) {
    if (!ctx || !x.ptr || !w.ptr || !bias.ptr || !out.ptr || x_rows <= 0 || cols <= 0 || out_cols <= 0) return -1;
    
    bool request_mps = getenv("MU_DENSE_ROWS_MPS") != NULL;
    bool use_simdgroup = mu_gpu_dense_mps_shape(cols, out_cols) &&
                         ctx->gpu->dense_bf16_bias_rows_simdgroup_quick_gelu &&
                         getenv("MU_DENSE_ROWS_NO_SIMDGROUP") == NULL;
                         
    if (use_simdgroup && !request_mps) {
        id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w.ptr;
        id<MTLBuffer> bias_buf = (__bridge id<MTLBuffer>)bias.ptr;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

        [ctx->encoder setComputePipelineState:ctx->gpu->dense_bf16_bias_rows_simdgroup_quick_gelu];
        [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
        [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
        [ctx->encoder setBuffer:bias_buf offset:bias.offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [ctx->encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:5];
        [ctx->encoder setBytes:&x_rows length:sizeof(x_rows) atIndex:6];

        MTLSize grid = MTLSizeMake(((NSUInteger)out_cols + 31u) / 32u, ((NSUInteger)x_rows + 7u) / 8u, 1);
        MTLSize threads = MTLSizeMake(32, 1, 1);
        [ctx->encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        return 0;
    }
    
    int rc = mu_gpu_dense_bf16_bias_rows_ctx(ctx, x, w, bias, x_rows, cols, out_cols, out);
    if (rc != 0) return rc;
    return mu_gpu_vision_quick_gelu_bf16_ctx(ctx, out, x_rows * out_cols, out);
}

int mu_gpu_dense_bf16_bias_rows_gelu_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                         mu_gpu_buf bias, int x_rows, int cols, int out_cols, mu_gpu_buf out) {
    if (!ctx || !x.ptr || !w.ptr || !bias.ptr || !out.ptr || x_rows <= 0 || cols <= 0 || out_cols <= 0) return -1;
    
    bool request_mps = getenv("MU_DENSE_ROWS_MPS") != NULL;
    bool use_simdgroup = mu_gpu_dense_mps_shape(cols, out_cols) &&
                         ctx->gpu->dense_bf16_bias_rows_simdgroup_gelu &&
                         getenv("MU_DENSE_ROWS_NO_SIMDGROUP") == NULL;
                         
    if (use_simdgroup && !request_mps) {
        id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w.ptr;
        id<MTLBuffer> bias_buf = (__bridge id<MTLBuffer>)bias.ptr;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

        [ctx->encoder setComputePipelineState:ctx->gpu->dense_bf16_bias_rows_simdgroup_gelu];
        [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
        [ctx->encoder setBuffer:w_buf offset:w.offset atIndex:1];
        [ctx->encoder setBuffer:bias_buf offset:bias.offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [ctx->encoder setBytes:&out_cols length:sizeof(out_cols) atIndex:5];
        [ctx->encoder setBytes:&x_rows length:sizeof(x_rows) atIndex:6];

        MTLSize grid = MTLSizeMake(((NSUInteger)out_cols + 31u) / 32u, ((NSUInteger)x_rows + 7u) / 8u, 1);
        MTLSize threads = MTLSizeMake(32, 1, 1);
        [ctx->encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        return 0;
    }
    
    int rc = mu_gpu_dense_bf16_bias_rows_ctx(ctx, x, w, bias, x_rows, cols, out_cols, out);
    if (rc != 0) return rc;
    return mu_gpu_vision_gelu_bf16_ctx(ctx, out, x_rows * out_cols, out);
}

int mu_gpu_text_decode_fused_ffn_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf hs_in,
                                     mu_gpu_buf post_norm_w, mu_gpu_buf gate_w,
                                     mu_gpu_buf up_w, mu_gpu_buf down_w,
                                     float eps, mu_gpu_buf out) {
    if (!ctx || !hs_in.ptr || !post_norm_w.ptr || !gate_w.ptr || !up_w.ptr || !down_w.ptr || !out.ptr) return -1;
    
    bool request_fallback = getenv("MU_TEXT_NO_FUSED_FFN") != NULL;
    if (!request_fallback && ctx->gpu->text_decode_fused_ffn) {
        id<MTLBuffer> hs_in_buf = (__bridge id<MTLBuffer>)hs_in.ptr;
        id<MTLBuffer> post_norm_w_buf = (__bridge id<MTLBuffer>)post_norm_w.ptr;
        id<MTLBuffer> gate_w_buf = (__bridge id<MTLBuffer>)gate_w.ptr;
        id<MTLBuffer> up_w_buf = (__bridge id<MTLBuffer>)up_w.ptr;
        id<MTLBuffer> down_w_buf = (__bridge id<MTLBuffer>)down_w.ptr;
        id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

        [ctx->encoder setComputePipelineState:ctx->gpu->text_decode_fused_ffn];
        [ctx->encoder setBuffer:hs_in_buf offset:hs_in.offset atIndex:0];
        [ctx->encoder setBuffer:post_norm_w_buf offset:post_norm_w.offset atIndex:1];
        [ctx->encoder setBuffer:gate_w_buf offset:gate_w.offset atIndex:2];
        [ctx->encoder setBuffer:up_w_buf offset:up_w.offset atIndex:3];
        [ctx->encoder setBuffer:down_w_buf offset:down_w.offset atIndex:4];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:5];
        [ctx->encoder setBytes:&eps length:sizeof(eps) atIndex:6];

        MTLSize grid = MTLSizeMake(1, 1, 1);
        MTLSize threads = MTLSizeMake(256, 1, 1);
        [ctx->encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        return 0;
    }
    
    NSUInteger base_offset_a = ctx->alloc.offset_a;
    NSUInteger base_offset_b = ctx->alloc.offset_b;
    
    mu_gpu_buf normed_buf = mu_gpu_scratch_alloc_a_ctx(ctx, 896 * sizeof(float));
    mu_gpu_buf gate_buf = mu_gpu_scratch_alloc_a_ctx(ctx, 4864 * sizeof(float));
    mu_gpu_buf up_buf = mu_gpu_scratch_alloc_a_ctx(ctx, 4864 * sizeof(float));
    mu_gpu_buf mid_buf = mu_gpu_scratch_alloc_a_ctx(ctx, 4864 * sizeof(float));
    mu_gpu_buf proj_buf = mu_gpu_scratch_alloc_a_ctx(ctx, 896 * sizeof(float));
    
    if (!normed_buf.ptr || !gate_buf.ptr || !up_buf.ptr || !mid_buf.ptr || !proj_buf.ptr) {
        return -2;
    }
    
    int rc = mu_gpu_rmsnorm_bf16_probe_ctx(ctx, hs_in, post_norm_w, normed_buf, 896, eps);
    if (rc == 0) rc = mu_gpu_dense_probe_ctx(ctx, normed_buf, gate_w, gate_buf, 4864, 896);
    if (rc == 0) rc = mu_gpu_dense_probe_ctx(ctx, normed_buf, up_w, up_buf, 4864, 896);
    if (rc == 0) rc = mu_gpu_silu_mul_f32_ctx(ctx, gate_buf, up_buf, mid_buf, 4864);
    if (rc == 0) rc = mu_gpu_dense_probe_ctx(ctx, mid_buf, down_w, proj_buf, 896, 4864);
    if (rc == 0) rc = mu_gpu_add_f32_ctx(ctx, hs_in, proj_buf, out, 896);
    
    ctx->alloc.offset_a = base_offset_a;
    ctx->alloc.offset_b = base_offset_b;
    return rc;
}

int mu_gpu_vision_attn_concat_probe_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q0, mu_gpu_buf kv,
                                        const float *rotary, int rows, int token_index, mu_gpu_buf out) {
    if (!ctx || !q0.ptr || !kv.ptr || !rotary || !out.ptr || rows <= 0 || token_index < 0) return -1;
    id<MTLBuffer> q0_buf = (__bridge id<MTLBuffer>)q0.ptr;
    id<MTLBuffer> kv_buf = (__bridge id<MTLBuffer>)kv.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    NSUInteger rotary_bytes = (NSUInteger)rows * 40u * sizeof(float);
    NSUInteger rotary_offset = 0;
    id<MTLBuffer> rotary_buf = mu_scratch_alloc_a(&ctx->alloc, rotary_bytes, &rotary_offset);
    if (rotary_buf) {
        memcpy((char *)[rotary_buf contents] + rotary_offset, rotary, rotary_bytes);
    } else {
        rotary_buf = [ctx->gpu->device newBufferWithBytes:rotary length:rotary_bytes options:MTLResourceStorageModeShared];
    }

    [ctx->encoder setComputePipelineState:ctx->gpu->vision_attn_concat_probe];
    [ctx->encoder setBuffer:q0_buf offset:q0.offset atIndex:0];
    [ctx->encoder setBuffer:kv_buf offset:kv.offset atIndex:1];
    [ctx->encoder setBuffer:rotary_buf offset:rotary_offset atIndex:2];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
    [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [ctx->encoder setBytes:&token_index length:sizeof(token_index) atIndex:5];

    NSUInteger width = ctx->gpu->vision_attn_concat_probe.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > 1280u) width = 1280u;
    MTLSize grid = MTLSizeMake(1280u, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_vision_attn_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q, mu_gpu_buf kv,
                                const float *rotary, int rows, mu_gpu_buf out) {
    if (!ctx || !q.ptr || !kv.ptr || !rotary || !out.ptr || rows <= 0) return -1;
    id<MTLBuffer> q_buf = (__bridge id<MTLBuffer>)q.ptr;
    id<MTLBuffer> kv_buf = (__bridge id<MTLBuffer>)kv.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    NSUInteger rotary_bytes = (NSUInteger)rows * 40u * sizeof(float);
    NSUInteger rotary_offset = 0;
    id<MTLBuffer> rotary_buf = mu_scratch_alloc_a(&ctx->alloc, rotary_bytes, &rotary_offset);
    if (rotary_buf) {
        memcpy((char *)[rotary_buf contents] + rotary_offset, rotary, rotary_bytes);
    } else {
        rotary_buf = [ctx->gpu->device newBufferWithBytes:rotary length:rotary_bytes options:MTLResourceStorageModeShared];
    }

    bool request_flash = getenv("MU_VISION_ATTN_NO_FLASH") == NULL;
    bool shape_profile = getenv("MU_VISION_ATTN_SHAPE_PROFILE") != NULL;
    bool request_flash_k16 = getenv("MU_VISION_ATTN_FLASH_K16") != NULL;
    bool use_flash_k16 = request_flash_k16 && ctx->gpu->vision_attn_rows_flash_k16 != nil;
    id<MTLComputePipelineState> flash_pipeline = use_flash_k16
        ? ctx->gpu->vision_attn_rows_flash_k16
        : ctx->gpu->vision_attn_rows_flash;
    if (request_flash && flash_pipeline) {
        [ctx->encoder setComputePipelineState:flash_pipeline];
        [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
        [ctx->encoder setBuffer:kv_buf offset:kv.offset atIndex:1];
        [ctx->encoder setBuffer:rotary_buf offset:rotary_offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        
        MTLSize grid = MTLSizeMake(((NSUInteger)rows + 31u) / 32u, 16u, 1);
        MTLSize threads = MTLSizeMake(32, 1, 1);
        if (shape_profile) {
            if (use_flash_k16) {
                fprintf(stderr,
                        "mu_profile stage=vision_attn_shape path=flash_k16 rows=%d "
                        "threadgroups=%lux%lux%lu threads=%lux%lux%lu "
                        "query_tile_rows=32 key_tile_rows=16 heads=16\n",
                        rows,
                        (unsigned long)grid.width,
                        (unsigned long)grid.height,
                        (unsigned long)grid.depth,
                        (unsigned long)threads.width,
                        (unsigned long)threads.height,
                        (unsigned long)threads.depth);
            } else {
                fprintf(stderr,
                        "mu_profile stage=vision_attn_shape path=flash rows=%d "
                        "threadgroups=%lux%lux%lu threads=%lux%lux%lu "
                        "query_tile_rows=32 key_tile_rows=32 heads=16\n",
                        rows,
                        (unsigned long)grid.width,
                        (unsigned long)grid.height,
                        (unsigned long)grid.depth,
                        (unsigned long)threads.width,
                        (unsigned long)threads.height,
                        (unsigned long)threads.depth);
            }
        }
        [ctx->encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        return 0;
    }

    if (getenv("MU_VISION_ATTN_ONLINE") != NULL && ctx->gpu->vision_attn_rows_online) {
        [ctx->encoder setComputePipelineState:ctx->gpu->vision_attn_rows_online];
        [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
        [ctx->encoder setBuffer:kv_buf offset:kv.offset atIndex:1];
        [ctx->encoder setBuffer:rotary_buf offset:rotary_offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        MTLSize grid = MTLSizeMake((NSUInteger)rows, 16u, 1);
        MTLSize threads = MTLSizeMake(1, 16u, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        return 0;
    }

    bool request_prerotate = getenv("MU_VISION_ATTN_PREROTATE") != NULL;
    bool disable_prerotate = getenv("MU_VISION_ATTN_NO_PREROTATE") != NULL;
    bool use_prerotate = (request_prerotate || !disable_prerotate) &&
                         ctx->gpu->vision_rope_qk_rows &&
                         ctx->gpu->vision_qk_scores_head_prerot;
    id<MTLBuffer> q_rot_buf = nil;
    id<MTLBuffer> k_rot_buf = nil;
    NSUInteger q_rot_offset = 0;
    NSUInteger k_rot_offset = 0;
    if (use_prerotate) {
        NSUInteger rot_bytes = (NSUInteger)rows * 1280u * sizeof(float);
        q_rot_buf = mu_scratch_alloc_a(&ctx->alloc, rot_bytes, &q_rot_offset);
        if (!q_rot_buf) {
            q_rot_buf = [ctx->gpu->device newBufferWithLength:rot_bytes
                                                      options:MTLResourceStorageModeShared];
        }
        k_rot_buf = mu_scratch_alloc_a(&ctx->alloc, rot_bytes, &k_rot_offset);
        if (!k_rot_buf) {
            k_rot_buf = [ctx->gpu->device newBufferWithLength:rot_bytes
                                                      options:MTLResourceStorageModeShared];
        }
        if (!q_rot_buf || !k_rot_buf) return -2;

        [ctx->encoder setComputePipelineState:ctx->gpu->vision_rope_qk_rows];
        [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
        [ctx->encoder setBuffer:kv_buf offset:kv.offset atIndex:1];
        [ctx->encoder setBuffer:rotary_buf offset:rotary_offset atIndex:2];
        [ctx->encoder setBuffer:q_rot_buf offset:q_rot_offset atIndex:3];
        [ctx->encoder setBuffer:k_rot_buf offset:k_rot_offset atIndex:4];
        [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        NSUInteger rope_width = ctx->gpu->vision_rope_qk_rows.threadExecutionWidth;
        if (rope_width < 1) rope_width = 1;
        if (rope_width > 256u) rope_width = 256u;
        [ctx->encoder dispatchThreads:MTLSizeMake((NSUInteger)rows * 1280u, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(rope_width, 1, 1)];
    }

    NSUInteger scores_bytes = (NSUInteger)rows * (NSUInteger)rows * sizeof(float);
    NSUInteger scores_offset = 0;
    id<MTLBuffer> scores_buf = mu_scratch_alloc_b(&ctx->alloc, scores_bytes, &scores_offset);
    if (!scores_buf) {
        scores_buf = [ctx->gpu->device newBufferWithLength:scores_bytes options:MTLResourceStorageModeShared];
    }

    for (int head = 0; head < 16; head++) {
        id<MTLComputePipelineState> qk_pipeline = use_prerotate
            ? ctx->gpu->vision_qk_scores_head_prerot
            : ctx->gpu->vision_qk_scores_head;
        [ctx->encoder setComputePipelineState:qk_pipeline];
        if (use_prerotate) {
            [ctx->encoder setBuffer:q_rot_buf offset:q_rot_offset atIndex:0];
            [ctx->encoder setBuffer:k_rot_buf offset:k_rot_offset atIndex:1];
            [ctx->encoder setBuffer:scores_buf offset:scores_offset atIndex:2];
            [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:3];
            [ctx->encoder setBytes:&head length:sizeof(head) atIndex:4];
        } else {
            [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
            [ctx->encoder setBuffer:kv_buf offset:kv.offset atIndex:1];
            [ctx->encoder setBuffer:rotary_buf offset:rotary_offset atIndex:2];
            [ctx->encoder setBuffer:scores_buf offset:scores_offset atIndex:3];
            [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:4];
            [ctx->encoder setBytes:&head length:sizeof(head) atIndex:5];
        }
        NSUInteger qk_width = qk_pipeline.threadExecutionWidth;
        if (qk_width < 1) qk_width = 1;
        if (qk_width > 16u) qk_width = 16u;
        [ctx->encoder dispatchThreads:MTLSizeMake((NSUInteger)rows, (NSUInteger)rows, 1)
               threadsPerThreadgroup:MTLSizeMake(qk_width, 1, 1)];

        bool request_fused_pv = getenv("MU_VISION_ATTN_FUSED_PV") != NULL;
        bool disable_fused_pv = getenv("MU_VISION_ATTN_NO_FUSED_PV") != NULL;
        bool use_fused_pv = (request_fused_pv || !disable_fused_pv) &&
                            ctx->gpu->vision_softmax_pv_head;
        if (use_fused_pv) {
            [ctx->encoder setComputePipelineState:ctx->gpu->vision_softmax_pv_head];
            [ctx->encoder setBuffer:scores_buf offset:scores_offset atIndex:0];
            [ctx->encoder setBuffer:kv_buf offset:kv.offset atIndex:1];
            [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
            [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:3];
            [ctx->encoder setBytes:&head length:sizeof(head) atIndex:4];
            NSUInteger fused_width = ctx->gpu->vision_softmax_pv_head.threadExecutionWidth;
            if (fused_width < 1) fused_width = 1;
            if (fused_width > (NSUInteger)rows) fused_width = (NSUInteger)rows;
            [ctx->encoder dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(fused_width, 1, 1)];
        } else {
            [ctx->encoder setComputePipelineState:ctx->gpu->vision_softmax_bf16_rows];
            [ctx->encoder setBuffer:scores_buf offset:scores_offset atIndex:0];
            [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:1];
            NSUInteger sm_width = ctx->gpu->vision_softmax_bf16_rows.threadExecutionWidth;
            if (sm_width < 1) sm_width = 1;
            if (sm_width > (NSUInteger)rows) sm_width = (NSUInteger)rows;
            [ctx->encoder dispatchThreads:MTLSizeMake((NSUInteger)rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(sm_width, 1, 1)];

            [ctx->encoder setComputePipelineState:ctx->gpu->vision_pv_head];
            [ctx->encoder setBuffer:scores_buf offset:scores_offset atIndex:0];
            [ctx->encoder setBuffer:kv_buf offset:kv.offset atIndex:1];
            [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
            [ctx->encoder setBytes:&rows length:sizeof(rows) atIndex:3];
            [ctx->encoder setBytes:&head length:sizeof(head) atIndex:4];
            NSUInteger pv_width = ctx->gpu->vision_pv_head.threadExecutionWidth;
            if (pv_width < 1) pv_width = 1;
            if (pv_width > 80u) pv_width = 80u;
            [ctx->encoder dispatchThreads:MTLSizeMake(80u, (NSUInteger)rows, 1)
                   threadsPerThreadgroup:MTLSizeMake(pv_width, 1, 1)];
        }
    }
    return 0;
}

int mu_gpu_vision_add_bf16_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf a, mu_gpu_buf b,
                               int n, mu_gpu_buf out) {
    if (!ctx || !a.ptr || !b.ptr || !out.ptr || n <= 0) return -1;
    id<MTLBuffer> a_buf = (__bridge id<MTLBuffer>)a.ptr;
    id<MTLBuffer> b_buf = (__bridge id<MTLBuffer>)b.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->vision_add_bf16];
    [ctx->encoder setBuffer:a_buf offset:a.offset atIndex:0];
    [ctx->encoder setBuffer:b_buf offset:b.offset atIndex:1];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:2];
    [ctx->encoder setBytes:&n length:sizeof(n) atIndex:3];

    NSUInteger width = ctx->gpu->vision_add_bf16.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)n) width = (NSUInteger)n;
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_vision_quick_gelu_bf16_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x,
                                      int n, mu_gpu_buf out) {
    if (!ctx || !x.ptr || !out.ptr || n <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->vision_quick_gelu_bf16];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:1];
    [ctx->encoder setBytes:&n length:sizeof(n) atIndex:2];

    NSUInteger width = ctx->gpu->vision_quick_gelu_bf16.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)n) width = (NSUInteger)n;
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_vision_gelu_bf16_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x,
                                 int n, mu_gpu_buf out) {
    if (!ctx || !x.ptr || !out.ptr || n <= 0) return -1;
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->vision_gelu_bf16];
    [ctx->encoder setBuffer:x_buf offset:x.offset atIndex:0];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:1];
    [ctx->encoder setBytes:&n length:sizeof(n) atIndex:2];

    NSUInteger width = ctx->gpu->vision_gelu_bf16.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)n) width = (NSUInteger)n;
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}

int mu_gpu_vision_merge4_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf hidden,
                             int rows, mu_gpu_buf out) {
    if (!ctx || !hidden.ptr || !out.ptr || rows <= 0) return -1;
    id<MTLBuffer> hidden_buf = (__bridge id<MTLBuffer>)hidden.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    int groups = rows / 4;
    [ctx->encoder setComputePipelineState:ctx->gpu->vision_merge4];
    [ctx->encoder setBuffer:hidden_buf offset:hidden.offset atIndex:0];
    [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:1];
    [ctx->encoder setBytes:&groups length:sizeof(groups) atIndex:2];

    int total = groups * 5120;
    NSUInteger width = ctx->gpu->vision_merge4.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > (NSUInteger)total) width = (NSUInteger)total;
    MTLSize grid = MTLSizeMake((NSUInteger)total, 1, 1);
    MTLSize threads = MTLSizeMake(width, 1, 1);
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    return 0;
}


int mu_gpu_kv_cache_create(mu_gpu *gpu, int layers, int cap, mu_gpu_kv_cache **out) {
    if (!gpu || layers <= 0 || cap <= 0 || !out) return -1;
    mu_gpu_kv_cache *cache = (mu_gpu_kv_cache *)calloc(1, sizeof(*cache));
    if (!cache) return -2;

    cache->gpu = gpu;
    cache->layers = layers;
    cache->cap = cap;

    NSUInteger bytes = (NSUInteger)layers * (NSUInteger)cap * 128u * sizeof(float);
    cache->k_cache = [gpu->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    cache->v_cache = [gpu->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    if (!cache->k_cache || !cache->v_cache) {
        cache->k_cache = nil;
        cache->v_cache = nil;
        free(cache);
        return -3;
    }

    *out = cache;
    return 0;
}

void mu_gpu_kv_cache_destroy(mu_gpu_kv_cache *cache) {
    if (!cache) return;
    cache->k_cache = nil;
    cache->v_cache = nil;
    free(cache);
}

int mu_gpu_kv_cache_update_layer(mu_gpu_kv_cache *cache, int layer, int pos, const float *k_val, const float *v_val) {
    if (!cache || layer < 0 || layer >= cache->layers || pos < 0 || pos >= cache->cap) return -1;

    size_t offset = ((size_t)layer * (size_t)cache->cap + (size_t)pos) * 128u;
    if (k_val) {
        float *k_ptr = (float *)[cache->k_cache contents] + offset;
        memcpy(k_ptr, k_val, 128u * sizeof(float));
    }
    if (v_val) {
        float *v_ptr = (float *)[cache->v_cache contents] + offset;
        memcpy(v_ptr, v_val, 128u * sizeof(float));
    }
    return 0;
}

int mu_gpu_kv_cache_upload_all(mu_gpu_kv_cache *cache, const float *k_cpu, const float *v_cpu) {
    if (!cache) return -1;
    NSUInteger bytes = (NSUInteger)cache->layers * (NSUInteger)cache->cap * 128u * sizeof(float);
    if (k_cpu) {
        memcpy([cache->k_cache contents], k_cpu, bytes);
    }
    if (v_cpu) {
        memcpy([cache->v_cache contents], v_cpu, bytes);
    }
    return 0;
}

int mu_gpu_text_rope_cache_update_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q,
                                      mu_gpu_buf k, mu_gpu_buf v,
                                      mu_gpu_kv_cache *cache, int layer,
                                      int cache_pos, const int pos3[3]) {
    if (!ctx || !q.ptr || !k.ptr || !v.ptr || !cache || !pos3 ||
        !ctx->gpu->text_rope_cache_update ||
        layer < 0 || layer >= cache->layers ||
        cache_pos < 0 || cache_pos >= cache->cap) {
        return -1;
    }
    id<MTLBuffer> q_buf = (__bridge id<MTLBuffer>)q.ptr;
    id<MTLBuffer> k_buf = (__bridge id<MTLBuffer>)k.ptr;
    id<MTLBuffer> v_buf = (__bridge id<MTLBuffer>)v.ptr;
    NSUInteger kv_offset = (NSUInteger)layer * (NSUInteger)cache->cap * 128u * sizeof(float);

    [ctx->encoder setComputePipelineState:ctx->gpu->text_rope_cache_update];
    [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
    [ctx->encoder setBuffer:k_buf offset:k.offset atIndex:1];
    [ctx->encoder setBuffer:v_buf offset:v.offset atIndex:2];
    [ctx->encoder setBuffer:cache->k_cache offset:kv_offset atIndex:3];
    [ctx->encoder setBuffer:cache->v_cache offset:kv_offset atIndex:4];
    [ctx->encoder setBytes:pos3 length:3 * sizeof(pos3[0]) atIndex:5];
    [ctx->encoder setBytes:&cache_pos length:sizeof(cache_pos) atIndex:6];

    MTLSize grid = MTLSizeMake(448, 1, 1);
    NSUInteger width = ctx->gpu->text_rope_cache_update.threadExecutionWidth;
    if (width < 1) width = 1;
    if (width > 448u) width = 448u;
    [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    return 0;
}

int mu_gpu_text_attn_cached_resident_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q,
                                         mu_gpu_kv_cache *cache, int layer,
                                         int cache_len, mu_gpu_buf out) {
    if (!ctx || !q.ptr || !cache || !out.ptr || cache_len <= 0 ||
        layer < 0 || layer >= cache->layers) return -1;
    id<MTLBuffer> q_buf = (__bridge id<MTLBuffer>)q.ptr;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out.ptr;

    NSUInteger kv_offset = (NSUInteger)layer * (NSUInteger)cache->cap * 128u * sizeof(float);

    bool disable_simd = getenv("MU_TEXT_ATTN_CACHED_NO_SIMD") != NULL;
    bool use_simd = !disable_simd;
    if (use_simd && ctx->gpu->text_attn_cached_simd) {
        [ctx->encoder setComputePipelineState:ctx->gpu->text_attn_cached_simd];
        [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
        [ctx->encoder setBuffer:cache->k_cache offset:kv_offset atIndex:1];
        [ctx->encoder setBuffer:cache->v_cache offset:kv_offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&cache_len length:sizeof(cache_len) atIndex:4];

        MTLSize grid = MTLSizeMake(14 * 32, 1, 1);
        MTLSize threads = MTLSizeMake(32, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    } else {
        [ctx->encoder setComputePipelineState:ctx->gpu->text_attn_cached];
        [ctx->encoder setBuffer:q_buf offset:q.offset atIndex:0];
        [ctx->encoder setBuffer:cache->k_cache offset:kv_offset atIndex:1];
        [ctx->encoder setBuffer:cache->v_cache offset:kv_offset atIndex:2];
        [ctx->encoder setBuffer:out_buf offset:out.offset atIndex:3];
        [ctx->encoder setBytes:&cache_len length:sizeof(cache_len) atIndex:4];

        MTLSize grid = MTLSizeMake(14, 1, 1);
        MTLSize threads = MTLSizeMake(1, 1, 1);
        [ctx->encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }
    return 0;
}

int mu_gpu_text_logits_argmax_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf hidden_state,
                                  mu_gpu_buf final_norm_bf16,
                                  mu_gpu_buf embed_bf16,
                                  float eps, int hidden_dim, int vocab_dim,
                                  mu_gpu_buf out_id, mu_gpu_buf out_val) {
    if (!ctx || !ctx->gpu || !ctx->encoder || !hidden_state.ptr ||
        !final_norm_bf16.ptr || !embed_bf16.ptr || !out_id.ptr ||
        !out_val.ptr || hidden_dim <= 0 || vocab_dim <= 0 ||
        !ctx->gpu->rmsnorm_bf16_rows || !ctx->gpu->dense_f32_rows ||
        !ctx->gpu->argmax_f32) {
        return -1;
    }

    mu_gpu_buf last = mu_gpu_scratch_alloc_a_ctx(ctx, (unsigned long)hidden_dim * sizeof(float));
    mu_gpu_buf logits = mu_gpu_scratch_alloc_b_ctx(ctx, (unsigned long)vocab_dim * sizeof(float));
    if (!last.ptr || !logits.ptr) return -2;

    id<MTLBuffer> hidden_buf = (__bridge id<MTLBuffer>)hidden_state.ptr;
    id<MTLBuffer> norm_buf = (__bridge id<MTLBuffer>)final_norm_bf16.ptr;
    id<MTLBuffer> last_buf = (__bridge id<MTLBuffer>)last.ptr;
    id<MTLBuffer> embed_buf = (__bridge id<MTLBuffer>)embed_bf16.ptr;
    id<MTLBuffer> logits_buf = (__bridge id<MTLBuffer>)logits.ptr;
    id<MTLBuffer> out_id_buf = (__bridge id<MTLBuffer>)out_id.ptr;
    id<MTLBuffer> out_val_buf = (__bridge id<MTLBuffer>)out_val.ptr;

    [ctx->encoder setComputePipelineState:ctx->gpu->rmsnorm_bf16_rows];
    [ctx->encoder setBuffer:hidden_buf offset:hidden_state.offset atIndex:0];
    [ctx->encoder setBuffer:norm_buf offset:final_norm_bf16.offset atIndex:1];
    [ctx->encoder setBuffer:last_buf offset:last.offset atIndex:2];
    [ctx->encoder setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:3];
    [ctx->encoder setBytes:&eps length:sizeof(eps) atIndex:4];

    NSUInteger w_norm = ctx->gpu->rmsnorm_bf16_rows.threadExecutionWidth;
    if (w_norm < 1) w_norm = 1;
    if (w_norm > (NSUInteger)hidden_dim) w_norm = (NSUInteger)hidden_dim;
    [ctx->encoder dispatchThreads:MTLSizeMake((NSUInteger)hidden_dim, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(w_norm, 1, 1)];

    [ctx->encoder setComputePipelineState:ctx->gpu->dense_f32_rows];
    [ctx->encoder setBuffer:last_buf offset:last.offset atIndex:0];
    [ctx->encoder setBuffer:embed_buf offset:embed_bf16.offset atIndex:1];
    [ctx->encoder setBuffer:logits_buf offset:logits.offset atIndex:2];
    [ctx->encoder setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:3];
    [ctx->encoder setBytes:&vocab_dim length:sizeof(vocab_dim) atIndex:4];

    NSUInteger w_dense = ctx->gpu->dense_f32_rows.threadExecutionWidth;
    if (w_dense < 1) w_dense = 1;
    if (w_dense > (NSUInteger)vocab_dim) w_dense = (NSUInteger)vocab_dim;
    [ctx->encoder dispatchThreads:MTLSizeMake((NSUInteger)vocab_dim, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(w_dense, 1, 1)];

    [ctx->encoder setComputePipelineState:ctx->gpu->argmax_f32];
    [ctx->encoder setBuffer:logits_buf offset:logits.offset atIndex:0];
    [ctx->encoder setBuffer:out_id_buf offset:out_id.offset atIndex:1];
    [ctx->encoder setBuffer:out_val_buf offset:out_val.offset atIndex:2];
    [ctx->encoder setBytes:&vocab_dim length:sizeof(vocab_dim) atIndex:3];
    [ctx->encoder dispatchThreads:MTLSizeMake(512, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

    return 0;
}

int mu_gpu_text_logits_argmax(mu_gpu *gpu, const float *hidden_state_cpu,
                              const unsigned short *final_norm_bf16,
                              const unsigned short *embed_bf16,
                              float eps, int hidden_dim, int vocab_dim,
                              int *out_id, float *out_val) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->rmsnorm_bf16_rows ||
        !gpu->dense_f32_rows || !gpu->argmax_f32) return -1;
    if (!hidden_state_cpu || !final_norm_bf16 || !embed_bf16 || !out_id || !out_val ||
        hidden_dim <= 0 || vocab_dim <= 0) return -2;

    @autoreleasepool {
        mu_scratch_allocator alloc_ctx = { gpu, 0, 0 };
        NSUInteger hs_bytes = (NSUInteger)hidden_dim * sizeof(float);
        NSUInteger norm_w_bytes = (NSUInteger)hidden_dim * sizeof(unsigned short);
        NSUInteger last_bytes = (NSUInteger)hidden_dim * sizeof(float);
        NSUInteger embed_bytes = (NSUInteger)vocab_dim * (NSUInteger)hidden_dim * sizeof(unsigned short);
        NSUInteger logits_bytes = (NSUInteger)vocab_dim * sizeof(float);

        NSUInteger hs_offset = 0;
        id<MTLBuffer> hs_buf = mu_scratch_alloc_a(&alloc_ctx, hs_bytes, &hs_offset);
        if (hs_buf) {
            memcpy((char *)[hs_buf contents] + hs_offset, hidden_state_cpu, hs_bytes);
        } else {
            hs_buf = [gpu->device newBufferWithBytes:hidden_state_cpu length:hs_bytes options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> norm_w_buf = mu_gpu_get_or_create_buffer(gpu, final_norm_bf16, norm_w_bytes);

        NSUInteger last_offset = 0;
        id<MTLBuffer> last_buf = mu_scratch_alloc_a(&alloc_ctx, last_bytes, &last_offset);
        if (!last_buf) {
            last_buf = [gpu->device newBufferWithLength:last_bytes options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> embed_buf = mu_gpu_get_or_create_buffer(gpu, embed_bf16, embed_bytes);

        NSUInteger logits_offset = 0;
        id<MTLBuffer> logits_buf = mu_scratch_alloc_b(&alloc_ctx, logits_bytes, &logits_offset);
        if (!logits_buf) {
            logits_buf = [gpu->device newBufferWithLength:logits_bytes options:MTLResourceStorageModeShared];
        }

        NSUInteger out_id_offset = 0;
        id<MTLBuffer> out_id_buf = mu_scratch_alloc_b(&alloc_ctx, sizeof(int), &out_id_offset);
        if (!out_id_buf) {
            out_id_buf = [gpu->device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
        }

        NSUInteger out_val_offset = 0;
        id<MTLBuffer> out_val_buf = mu_scratch_alloc_b(&alloc_ctx, sizeof(float), &out_val_offset);
        if (!out_val_buf) {
            out_val_buf = [gpu->device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        }

        if (!hs_buf || !norm_w_buf || !last_buf || !embed_buf || !logits_buf || !out_id_buf || !out_val_buf) {
            return -3;
        }

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        // 1. RMSNorm
        [encoder setComputePipelineState:gpu->rmsnorm_bf16_rows];
        [encoder setBuffer:hs_buf offset:hs_offset atIndex:0];
        [encoder setBuffer:norm_w_buf offset:0 atIndex:1];
        [encoder setBuffer:last_buf offset:last_offset atIndex:2];
        [encoder setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:3];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:4];

        NSUInteger w_norm = gpu->rmsnorm_bf16_rows.threadExecutionWidth;
        if (w_norm < 1) w_norm = 1;
        if (w_norm > (NSUInteger)hidden_dim) w_norm = (NSUInteger)hidden_dim;
        MTLSize grid_norm = MTLSizeMake((NSUInteger)hidden_dim, 1, 1);
        MTLSize threads_norm = MTLSizeMake(w_norm, 1, 1);
        [encoder dispatchThreads:grid_norm threadsPerThreadgroup:threads_norm];

        // 2. Dense
        [encoder setComputePipelineState:gpu->dense_f32_rows];
        [encoder setBuffer:last_buf offset:last_offset atIndex:0];
        [encoder setBuffer:embed_buf offset:0 atIndex:1];
        [encoder setBuffer:logits_buf offset:logits_offset atIndex:2];
        [encoder setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:3];
        [encoder setBytes:&vocab_dim length:sizeof(vocab_dim) atIndex:4];

        NSUInteger w_dense = gpu->dense_f32_rows.threadExecutionWidth;
        if (w_dense < 1) w_dense = 1;
        if (w_dense > (NSUInteger)vocab_dim) w_dense = (NSUInteger)vocab_dim;
        MTLSize grid_dense = MTLSizeMake((NSUInteger)vocab_dim, 1, 1);
        MTLSize threads_dense = MTLSizeMake(w_dense, 1, 1);
        [encoder dispatchThreads:grid_dense threadsPerThreadgroup:threads_dense];

        // 3. Argmax
        [encoder setComputePipelineState:gpu->argmax_f32];
        [encoder setBuffer:logits_buf offset:logits_offset atIndex:0];
        [encoder setBuffer:out_id_buf offset:out_id_offset atIndex:1];
        [encoder setBuffer:out_val_buf offset:out_val_offset atIndex:2];
        [encoder setBytes:&vocab_dim length:sizeof(vocab_dim) atIndex:3];

        MTLSize grid_argmax = MTLSizeMake(512, 1, 1);
        MTLSize threads_argmax = MTLSizeMake(512, 1, 1);
        [encoder dispatchThreads:grid_argmax threadsPerThreadgroup:threads_argmax];

        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out_id, (char *)[out_id_buf contents] + out_id_offset, sizeof(int));
        memcpy(out_val, (char *)[out_val_buf contents] + out_val_offset, sizeof(float));
    }
    return 0;
}
