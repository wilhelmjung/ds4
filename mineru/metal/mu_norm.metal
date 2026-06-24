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

kernel void mu_rmsnorm_bf16_probe(device const float *x [[buffer(0)]],
                                  device const ushort *weight [[buffer(1)]],
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
        out[gid] = x[gid] * scale * mu_bf16_to_f32(weight[gid]);
    }
}

kernel void mu_rmsnorm_bf16_rows(device const float *x [[buffer(0)]],
                                 device const ushort *weight [[buffer(1)]],
                                 device float *out [[buffer(2)]],
                                 constant int &cols [[buffer(3)]],
                                 constant float &eps [[buffer(4)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    int col = (int)gid.x;
    int row = (int)gid.y;
    if (col >= cols) return;

    device const float *xr = x + (size_t)row * (size_t)cols;
    float ss = 0.0f;
    for (int i = 0; i < cols; i++) {
        ss += xr[i] * xr[i];
    }
    float scale = rsqrt(ss / (float)cols + eps);
    out[(size_t)row * (size_t)cols + (size_t)col] =
        xr[col] * scale * mu_bf16_to_f32(weight[col]);
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

kernel void mu_layernorm_bf16_rows_simd(device const float *x [[buffer(0)]],
                                        device const ushort *weight [[buffer(1)]],
                                        device const ushort *bias [[buffer(2)]],
                                        device float *out [[buffer(3)]],
                                        constant int &cols [[buffer(4)]],
                                        constant float &eps [[buffer(5)]],
                                        uint row [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    device const float *xr = x + (size_t)row * (size_t)cols;

    float local = 0.0f;
    for (int i = (int)tid; i < cols; i += 256) {
        local += xr[i];
    }
    sums[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 128) sums[tid] += sums[tid + 128];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) sums[tid] += sums[tid + 64];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float s = sums[tid] + sums[tid + 32];
        s = simd_sum(s);
        if (tid == 0) sums[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = sums[0] / (float)cols;
    local = 0.0f;
    for (int i = (int)tid; i < cols; i += 256) {
        float d = xr[i] - mean;
        local += d * d;
    }
    sums[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 128) sums[tid] += sums[tid + 128];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) sums[tid] += sums[tid + 64];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float s = sums[tid] + sums[tid + 32];
        s = simd_sum(s);
        if (tid == 0) sums[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv = rsqrt(sums[0] / (float)cols + eps);
    for (int col = (int)tid; col < cols; col += 256) {
        float y = (xr[col] - mean) * inv;
        y = y * mu_bf16_to_f32(weight[col]) + mu_bf16_to_f32(bias[col]);
        out[(size_t)row * (size_t)cols + (size_t)col] = mu_round_bf16(y);
    }
}
