#include <metal_stdlib>
using namespace metal;

static inline float mu_bf16_to_f32(ushort v) {
    uint bits = ((uint)v) << 16;
    return as_type<float>(bits);
}

static inline ushort mu_f32_to_bf16_bits(float x) {
    uint bits = as_type<uint>(x);
    uint lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (ushort)(bits >> 16);
}

static inline float mu_round_bf16(float x) {
    return mu_bf16_to_f32(mu_f32_to_bf16_bits(x));
}

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

kernel void mu_layernorm_bf16_probe(device const float *x [[buffer(0)]],
                                    device const ushort *weight [[buffer(1)]],
                                    device const ushort *bias [[buffer(2)]],
                                    device float *out [[buffer(3)]],
                                    constant int &n [[buffer(4)]],
                                    constant float &eps [[buffer(5)]],
                                    uint gid [[thread_position_in_grid]]) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= (float)n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv = rsqrt(var / (float)n + eps);
    if ((int)gid < n) {
        float y = (x[gid] - mean) * inv;
        y = y * mu_bf16_to_f32(weight[gid]) + mu_bf16_to_f32(bias[gid]);
        out[gid] = mu_round_bf16(y);
    }
}

kernel void mu_layernorm_bf16_rows(device const float *x [[buffer(0)]],
                                   device const ushort *weight [[buffer(1)]],
                                   device const ushort *bias [[buffer(2)]],
                                   device float *out [[buffer(3)]],
                                   constant int &cols [[buffer(4)]],
                                   constant float &eps [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    int col = (int)gid.x;
    int row = (int)gid.y;
    if (col >= cols) return;

    device const float *xr = x + (size_t)row * (size_t)cols;
    float mean = 0.0f;
    for (int i = 0; i < cols; i++) {
        mean += xr[i];
    }
    mean /= (float)cols;

    float var = 0.0f;
    for (int i = 0; i < cols; i++) {
        float d = xr[i] - mean;
        var += d * d;
    }
    float inv = rsqrt(var / (float)cols + eps);
    float y = (xr[col] - mean) * inv;
    y = y * mu_bf16_to_f32(weight[col]) + mu_bf16_to_f32(bias[col]);
    out[(size_t)row * (size_t)cols + col] = mu_round_bf16(y);
}
