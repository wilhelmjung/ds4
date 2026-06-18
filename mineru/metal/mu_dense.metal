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

kernel void mu_dense_bf16_bias_probe(device const float *x [[buffer(0)]],
                                     device const ushort *w [[buffer(1)]],
                                     device const ushort *bias [[buffer(2)]],
                                     device float *out [[buffer(3)]],
                                     constant int &cols [[buffer(4)]],
                                     uint row [[thread_position_in_grid]]) {
    float acc = mu_bf16_to_f32(bias[row]);
    for (int c = 0; c < cols; c++) {
        acc += x[c] * mu_bf16_to_f32(w[row * cols + c]);
    }
    out[row] = mu_round_bf16(acc);
}

kernel void mu_dense_f32_bias_probe(device const float *x [[buffer(0)]],
                                    device const ushort *w [[buffer(1)]],
                                    device const ushort *bias [[buffer(2)]],
                                    device float *out [[buffer(3)]],
                                    constant int &cols [[buffer(4)]],
                                    uint row [[thread_position_in_grid]]) {
    float acc = mu_bf16_to_f32(bias[row]);
    for (int c = 0; c < cols; c++) {
        acc += x[c] * mu_bf16_to_f32(w[row * cols + c]);
    }
    out[row] = acc;
}

kernel void mu_dense_f32_rows(device const float *x [[buffer(0)]],
                              device const ushort *w [[buffer(1)]],
                              device float *out [[buffer(2)]],
                              constant int &cols [[buffer(3)]],
                              constant int &out_cols [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]]) {
    int out_col = (int)gid.x;
    int row = (int)gid.y;
    if (out_col >= out_cols) return;

    device const float *xrow = x + (size_t)row * (size_t)cols;
    device const ushort *wrow = w + (size_t)out_col * (size_t)cols;
    float acc = 0.0f;
    for (int c = 0; c < cols; c++) {
        acc += xrow[c] * mu_bf16_to_f32(wrow[c]);
    }
    out[(size_t)row * (size_t)out_cols + (size_t)out_col] = acc;
}

kernel void mu_dense_f32_bias_rows(device const float *x [[buffer(0)]],
                                   device const ushort *w [[buffer(1)]],
                                   device const ushort *bias [[buffer(2)]],
                                   device float *out [[buffer(3)]],
                                   constant int &cols [[buffer(4)]],
                                   constant int &out_cols [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    int out_col = (int)gid.x;
    int row = (int)gid.y;
    if (out_col >= out_cols) return;

    device const float *xrow = x + (size_t)row * (size_t)cols;
    device const ushort *wrow = w + (size_t)out_col * (size_t)cols;
    float acc = mu_bf16_to_f32(bias[out_col]);
    for (int c = 0; c < cols; c++) {
        acc += xrow[c] * mu_bf16_to_f32(wrow[c]);
    }
    out[(size_t)row * (size_t)out_cols + (size_t)out_col] = acc;
}

kernel void mu_dense_bf16_bias_rows(device const float *x [[buffer(0)]],
                                    device const ushort *w [[buffer(1)]],
                                    device const ushort *bias [[buffer(2)]],
                                    device float *out [[buffer(3)]],
                                    constant int &cols [[buffer(4)]],
                                    constant int &out_cols [[buffer(5)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    int out_col = (int)gid.x;
    int row = (int)gid.y;
    if (out_col >= out_cols) return;

    device const float *xrow = x + (size_t)row * (size_t)cols;
    device const ushort *wrow = w + (size_t)out_col * (size_t)cols;
    float acc = mu_bf16_to_f32(bias[out_col]);
    for (int c = 0; c < cols; c++) {
        acc += xrow[c] * mu_bf16_to_f32(wrow[c]);
    }
    out[(size_t)row * (size_t)out_cols + (size_t)out_col] = mu_round_bf16(acc);
}
