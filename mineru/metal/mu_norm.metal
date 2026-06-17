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
