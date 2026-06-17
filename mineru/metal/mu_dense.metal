#include <metal_stdlib>
using namespace metal;

static inline float mu_bf16_to_f32(ushort v) {
    uint bits = ((uint)v) << 16;
    return as_type<float>(bits);
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
