#include "mu_gpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

struct mu_gpu {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> dense_probe;
    char device_name[256];
};

static NSString *mu_gpu_shader_path(void) {
    NSString *cwd_path = @"mineru/metal/mu_dense.metal";
    if ([[NSFileManager defaultManager] fileExistsAtPath:cwd_path]) return cwd_path;

    NSString *src = [NSString stringWithUTF8String:__FILE__];
    NSString *dir = [src stringByDeletingLastPathComponent];
    NSString *from_src = [dir stringByAppendingPathComponent:@"metal/mu_dense.metal"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:from_src]) return from_src;

    return cwd_path;
}

static id<MTLComputePipelineState> mu_gpu_make_pipeline(id<MTLDevice> device,
                                                        NSString *function_name) {
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:mu_gpu_shader_path()
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
        gpu->dense_probe = mu_gpu_make_pipeline(device, @"mu_dense_probe");
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
