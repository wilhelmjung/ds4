#include "mu_gpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

struct mu_gpu {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> dense_probe;
    id<MTLComputePipelineState> dense_bf16_bias_probe;
    id<MTLComputePipelineState> rmsnorm_probe;
    id<MTLComputePipelineState> layernorm_bf16_probe;
    id<MTLComputePipelineState> layernorm_bf16_rows;
    id<MTLComputePipelineState> vision_attn_concat_probe;
    char device_name[256];
};

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
    if (!source) return nil;

    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) return nil;

    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (!function) return nil;

    return [device newComputePipelineStateWithFunction:function error:&error];
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
        gpu->rmsnorm_probe = mu_gpu_make_pipeline(device, @"mu_norm.metal",
                                                  @"mu_rmsnorm_probe");
        gpu->layernorm_bf16_probe = mu_gpu_make_pipeline(device, @"mu_norm.metal",
                                                         @"mu_layernorm_bf16_probe");
        gpu->layernorm_bf16_rows = mu_gpu_make_pipeline(device, @"mu_norm.metal",
                                                        @"mu_layernorm_bf16_rows");
        gpu->vision_attn_concat_probe = mu_gpu_make_pipeline(device, @"mu_vision.metal",
                                                             @"mu_vision_attn_concat_probe");
        const char *name = [[device name] UTF8String];
        if (name) {
            strlcpy(gpu->device_name, name, sizeof(gpu->device_name));
        } else {
            strlcpy(gpu->device_name, "unknown", sizeof(gpu->device_name));
        }
        *out = gpu;
    }
    return 0;
}

void mu_gpu_destroy(mu_gpu *gpu) {
    if (!gpu) return;
    gpu->vision_attn_concat_probe = nil;
    gpu->layernorm_bf16_rows = nil;
    gpu->layernorm_bf16_probe = nil;
    gpu->rmsnorm_probe = nil;
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
    if (!gpu || !gpu->device || !gpu->queue || !gpu->dense_probe) return -1;
    if (!x || !w_bf16 || !out || rows <= 0 || cols <= 0) return -2;

    @autoreleasepool {
        NSUInteger x_bytes = (NSUInteger)cols * sizeof(float);
        NSUInteger w_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(unsigned short);
        NSUInteger out_bytes = (NSUInteger)rows * sizeof(float);

        id<MTLBuffer> x_buf = [gpu->device newBufferWithBytes:x
                                                       length:x_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buf = [gpu->device newBufferWithBytes:w_bf16
                                                       length:w_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [gpu->device newBufferWithLength:out_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> cols_buf = [gpu->device newBufferWithBytes:&cols
                                                          length:sizeof(cols)
                                                         options:MTLResourceStorageModeShared];
        if (!x_buf || !w_buf || !out_buf || !cols_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->dense_probe];
        [encoder setBuffer:x_buf offset:0 atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBuffer:cols_buf offset:0 atIndex:3];

        NSUInteger width = gpu->dense_probe.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)rows) width = (NSUInteger)rows;
        MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, [out_buf contents], out_bytes);
    }
    return 0;
}

int mu_gpu_dense_bf16_bias_probe(mu_gpu *gpu, const float *x,
                                 const unsigned short *w_bf16,
                                 const unsigned short *bias_bf16,
                                 int rows, int cols, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->dense_bf16_bias_probe) return -1;
    if (!x || !w_bf16 || !bias_bf16 || !out || rows <= 0 || cols <= 0) return -2;

    @autoreleasepool {
        NSUInteger x_bytes = (NSUInteger)cols * sizeof(float);
        NSUInteger w_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(unsigned short);
        NSUInteger bias_bytes = (NSUInteger)rows * sizeof(unsigned short);
        NSUInteger out_bytes = (NSUInteger)rows * sizeof(float);

        id<MTLBuffer> x_buf = [gpu->device newBufferWithBytes:x
                                                       length:x_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buf = [gpu->device newBufferWithBytes:w_bf16
                                                       length:w_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bias_buf = [gpu->device newBufferWithBytes:bias_bf16
                                                          length:bias_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [gpu->device newBufferWithLength:out_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> cols_buf = [gpu->device newBufferWithBytes:&cols
                                                          length:sizeof(cols)
                                                         options:MTLResourceStorageModeShared];
        if (!x_buf || !w_buf || !bias_buf || !out_buf || !cols_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->dense_bf16_bias_probe];
        [encoder setBuffer:x_buf offset:0 atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:bias_buf offset:0 atIndex:2];
        [encoder setBuffer:out_buf offset:0 atIndex:3];
        [encoder setBuffer:cols_buf offset:0 atIndex:4];

        NSUInteger width = gpu->dense_bf16_bias_probe.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)rows) width = (NSUInteger)rows;
        MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, [out_buf contents], out_bytes);
    }
    return 0;
}

int mu_gpu_rmsnorm_probe(mu_gpu *gpu, const float *x, const float *weight,
                         int n, float eps, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->rmsnorm_probe) return -1;
    if (!x || !weight || !out || n <= 0) return -2;

    @autoreleasepool {
        NSUInteger bytes = (NSUInteger)n * sizeof(float);
        id<MTLBuffer> x_buf = [gpu->device newBufferWithBytes:x
                                                       length:bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buf = [gpu->device newBufferWithBytes:weight
                                                       length:bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [gpu->device newBufferWithLength:bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_buf = [gpu->device newBufferWithBytes:&n
                                                       length:sizeof(n)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> eps_buf = [gpu->device newBufferWithBytes:&eps
                                                         length:sizeof(eps)
                                                        options:MTLResourceStorageModeShared];
        if (!x_buf || !w_buf || !out_buf || !n_buf || !eps_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->rmsnorm_probe];
        [encoder setBuffer:x_buf offset:0 atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBuffer:n_buf offset:0 atIndex:3];
        [encoder setBuffer:eps_buf offset:0 atIndex:4];

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

        memcpy(out, [out_buf contents], bytes);
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
        NSUInteger x_bytes = (NSUInteger)n * sizeof(float);
        NSUInteger bf16_bytes = (NSUInteger)n * sizeof(unsigned short);
        id<MTLBuffer> x_buf = [gpu->device newBufferWithBytes:x
                                                       length:x_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buf = [gpu->device newBufferWithBytes:weight_bf16
                                                       length:bf16_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [gpu->device newBufferWithBytes:bias_bf16
                                                       length:bf16_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [gpu->device newBufferWithLength:x_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_buf = [gpu->device newBufferWithBytes:&n
                                                       length:sizeof(n)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> eps_buf = [gpu->device newBufferWithBytes:&eps
                                                         length:sizeof(eps)
                                                        options:MTLResourceStorageModeShared];
        if (!x_buf || !w_buf || !b_buf || !out_buf || !n_buf || !eps_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->layernorm_bf16_probe];
        [encoder setBuffer:x_buf offset:0 atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:b_buf offset:0 atIndex:2];
        [encoder setBuffer:out_buf offset:0 atIndex:3];
        [encoder setBuffer:n_buf offset:0 atIndex:4];
        [encoder setBuffer:eps_buf offset:0 atIndex:5];

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

        memcpy(out, [out_buf contents], x_bytes);
    }
    return 0;
}

int mu_gpu_layernorm_bf16_rows(mu_gpu *gpu, const float *x,
                               const unsigned short *weight_bf16,
                               const unsigned short *bias_bf16,
                               int rows, int cols, float eps, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->layernorm_bf16_rows) return -1;
    if (!x || !weight_bf16 || !bias_bf16 || !out || rows <= 0 || cols <= 0) return -2;

    @autoreleasepool {
        NSUInteger x_bytes = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
        NSUInteger bf16_bytes = (NSUInteger)cols * sizeof(unsigned short);
        id<MTLBuffer> x_buf = [gpu->device newBufferWithBytes:x
                                                       length:x_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buf = [gpu->device newBufferWithBytes:weight_bf16
                                                       length:bf16_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_buf = [gpu->device newBufferWithBytes:bias_bf16
                                                       length:bf16_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [gpu->device newBufferWithLength:x_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> cols_buf = [gpu->device newBufferWithBytes:&cols
                                                          length:sizeof(cols)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> eps_buf = [gpu->device newBufferWithBytes:&eps
                                                         length:sizeof(eps)
                                                        options:MTLResourceStorageModeShared];
        if (!x_buf || !w_buf || !b_buf || !out_buf || !cols_buf || !eps_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->layernorm_bf16_rows];
        [encoder setBuffer:x_buf offset:0 atIndex:0];
        [encoder setBuffer:w_buf offset:0 atIndex:1];
        [encoder setBuffer:b_buf offset:0 atIndex:2];
        [encoder setBuffer:out_buf offset:0 atIndex:3];
        [encoder setBuffer:cols_buf offset:0 atIndex:4];
        [encoder setBuffer:eps_buf offset:0 atIndex:5];

        NSUInteger width = gpu->layernorm_bf16_rows.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > (NSUInteger)cols) width = (NSUInteger)cols;
        MTLSize grid = MTLSizeMake((NSUInteger)cols, (NSUInteger)rows, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, [out_buf contents], x_bytes);
    }
    return 0;
}

int mu_gpu_vision_attn_concat_probe(mu_gpu *gpu, const float *q0,
                                    const float *kv, const float *rotary,
                                    int rows, int token_index, float *out) {
    if (!gpu || !gpu->device || !gpu->queue || !gpu->vision_attn_concat_probe) return -1;
    if (!q0 || !kv || !rotary || !out || rows <= 0 || token_index < 0 || token_index >= rows) {
        return -2;
    }

    @autoreleasepool {
        NSUInteger q_bytes = 1280u * sizeof(float);
        NSUInteger kv_bytes = (NSUInteger)rows * 2560u * sizeof(float);
        NSUInteger rotary_bytes = (NSUInteger)rows * 40u * sizeof(float);
        NSUInteger out_bytes = 1280u * sizeof(float);
        id<MTLBuffer> q_buf = [gpu->device newBufferWithBytes:q0
                                                       length:q_bytes
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> kv_buf = [gpu->device newBufferWithBytes:kv
                                                        length:kv_bytes
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> rotary_buf = [gpu->device newBufferWithBytes:rotary
                                                            length:rotary_bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [gpu->device newBufferWithLength:out_bytes
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> rows_buf = [gpu->device newBufferWithBytes:&rows
                                                          length:sizeof(rows)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> token_buf = [gpu->device newBufferWithBytes:&token_index
                                                           length:sizeof(token_index)
                                                          options:MTLResourceStorageModeShared];
        if (!q_buf || !kv_buf || !rotary_buf || !out_buf || !rows_buf || !token_buf) return -3;

        id<MTLCommandBuffer> command_buffer = [gpu->queue commandBuffer];
        if (!command_buffer) return -4;
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) return -5;

        [encoder setComputePipelineState:gpu->vision_attn_concat_probe];
        [encoder setBuffer:q_buf offset:0 atIndex:0];
        [encoder setBuffer:kv_buf offset:0 atIndex:1];
        [encoder setBuffer:rotary_buf offset:0 atIndex:2];
        [encoder setBuffer:out_buf offset:0 atIndex:3];
        [encoder setBuffer:rows_buf offset:0 atIndex:4];
        [encoder setBuffer:token_buf offset:0 atIndex:5];

        NSUInteger width = gpu->vision_attn_concat_probe.threadExecutionWidth;
        if (width < 1) width = 1;
        if (width > 1280u) width = 1280u;
        MTLSize grid = MTLSizeMake(1280u, 1, 1);
        MTLSize threads = MTLSizeMake(width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) return -6;

        memcpy(out, [out_buf contents], out_bytes);
    }
    return 0;
}

int mu_gpu_vision_encode(mu_gpu *gpu, void *engine,
                         const float *patch_embeds,
                         int rows, int cols,
                         const float *rotary,
                         int rotary_rows, int rotary_cols,
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
