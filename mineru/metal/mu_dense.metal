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
