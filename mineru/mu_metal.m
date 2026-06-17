#include "mu_gpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

struct mu_gpu {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    char device_name[256];
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
