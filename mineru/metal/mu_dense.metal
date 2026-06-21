#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

kernel void mu_dense_mps_bias_round(device float *out [[buffer(0)]],
                                    device const ushort *bias [[buffer(1)]],
                                    constant int &out_cols [[buffer(2)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    int out_col = (int)gid.x;
    int row = (int)gid.y;
    if (out_col >= out_cols) return;

    size_t idx = (size_t)row * (size_t)out_cols + (size_t)out_col;
    out[idx] = mu_round_bf16(out[idx] + mu_bf16_to_f32(bias[out_col]));
}

kernel void mu_dense_bf16_bias_rows_simd(device const float *x [[buffer(0)]],
                                         device const ushort *w [[buffer(1)]],
                                         device const ushort *bias [[buffer(2)]],
                                         device float *out [[buffer(3)]],
                                         constant int &cols [[buffer(4)]],
                                         constant int &out_cols [[buffer(5)]],
                                         uint2 gid [[thread_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    int out_col = (int)(gid.x / 32);
    int row = (int)gid.y;
    if (out_col >= out_cols) return;

    device const float *xrow = x + (size_t)row * (size_t)cols;
    device const ushort *wrow = w + (size_t)out_col * (size_t)cols;
    float partial = 0.0f;
    for (int c = (int)lane; c < cols; c += 32) {
        partial += xrow[c] * mu_bf16_to_f32(wrow[c]);
    }
    float acc = simd_sum(partial);
    if (lane == 0) {
        acc += mu_bf16_to_f32(bias[out_col]);
        out[(size_t)row * (size_t)out_cols + (size_t)out_col] =
            mu_round_bf16(acc);
    }
}

kernel void mu_dense_bf16_bias_rows_tiled(device const float *x [[buffer(0)]],
                                          device const ushort *w [[buffer(1)]],
                                          device const ushort *bias [[buffer(2)]],
                                          device float *out [[buffer(3)]],
                                          constant int &cols [[buffer(4)]],
                                          constant int &out_cols [[buffer(5)]],
                                          uint2 tg [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint sg [[simdgroup_index_in_threadgroup]]) {
    int out_col = (int)tg.x * 8 + (int)sg;
    int row = (int)tg.y;
    if (out_col >= out_cols) return;

    device const float *xrow = x + (size_t)row * (size_t)cols;
    device const ushort *wrow = w + (size_t)out_col * (size_t)cols;
    float partial = 0.0f;
    for (int c = (int)lane; c < cols; c += 32) {
        partial += xrow[c] * mu_bf16_to_f32(wrow[c]);
    }
    float acc = simd_sum(partial);
    if (lane == 0) {
        acc += mu_bf16_to_f32(bias[out_col]);
        out[(size_t)row * (size_t)out_cols + (size_t)out_col] =
            mu_round_bf16(acc);
    }
}

kernel void mu_dense_probe_simd(device const float *x [[buffer(0)]],
                                device const ushort *w [[buffer(1)]],
                                device float *out [[buffer(2)]],
                                constant int &cols [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]],
                                uint simd_lane [[thread_index_in_simdgroup]]) {
    uint row = gid.y;
    if ((cols & 3) == 0) {
        device const float4 *x_vec = (device const float4 *)x;
        device const ushort4 *w_vec = (device const ushort4 *)(w + (size_t)row * (size_t)cols);
        int cols_vec = cols / 4;
        float local_sum = 0.0f;
        for (int c = (int)simd_lane; c < cols_vec; c += 32) {
            float4 xv = x_vec[c];
            ushort4 wv = w_vec[c];
            local_sum += xv.x * mu_bf16_to_f32(wv.x) +
                         xv.y * mu_bf16_to_f32(wv.y) +
                         xv.z * mu_bf16_to_f32(wv.z) +
                         xv.w * mu_bf16_to_f32(wv.w);
        }
        float total_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            out[row] = total_sum;
        }
    } else {
        float local_sum = 0.0f;
        for (int c = (int)simd_lane; c < cols; c += 32) {
            local_sum += x[c] * mu_bf16_to_f32(w[(size_t)row * (size_t)cols + c]);
        }
        float total_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            out[row] = total_sum;
        }
    }
}

kernel void mu_dense_bf16_bias_probe_simd(device const float *x [[buffer(0)]],
                                          device const ushort *w [[buffer(1)]],
                                          device const ushort *bias [[buffer(2)]],
                                          device float *out [[buffer(3)]],
                                          constant int &cols [[buffer(4)]],
                                          uint2 gid [[thread_position_in_grid]],
                                          uint simd_lane [[thread_index_in_simdgroup]]) {
    uint row = gid.y;
    if ((cols & 3) == 0) {
        device const float4 *x_vec = (device const float4 *)x;
        device const ushort4 *w_vec = (device const ushort4 *)(w + (size_t)row * (size_t)cols);
        int cols_vec = cols / 4;
        float local_sum = 0.0f;
        for (int c = (int)simd_lane; c < cols_vec; c += 32) {
            float4 xv = x_vec[c];
            ushort4 wv = w_vec[c];
            local_sum += xv.x * mu_bf16_to_f32(wv.x) +
                         xv.y * mu_bf16_to_f32(wv.y) +
                         xv.z * mu_bf16_to_f32(wv.z) +
                         xv.w * mu_bf16_to_f32(wv.w);
        }
        float total_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            float bias_val = mu_bf16_to_f32(bias[row]);
            out[row] = mu_round_bf16(total_sum + bias_val);
        }
    } else {
        float local_sum = 0.0f;
        for (int c = (int)simd_lane; c < cols; c += 32) {
            local_sum += x[c] * mu_bf16_to_f32(w[(size_t)row * (size_t)cols + c]);
        }
        float total_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            float bias_val = mu_bf16_to_f32(bias[row]);
            out[row] = mu_round_bf16(total_sum + bias_val);
        }
    }
}

kernel void mu_dense_f32_bias_probe_simd(device const float *x [[buffer(0)]],
                                         device const ushort *w [[buffer(1)]],
                                         device const ushort *bias [[buffer(2)]],
                                         device float *out [[buffer(3)]],
                                         constant int &cols [[buffer(4)]],
                                         uint2 gid [[thread_position_in_grid]],
                                         uint simd_lane [[thread_index_in_simdgroup]]) {
    uint row = gid.y;
    if ((cols & 3) == 0) {
        device const float4 *x_vec = (device const float4 *)x;
        device const ushort4 *w_vec = (device const ushort4 *)(w + (size_t)row * (size_t)cols);
        int cols_vec = cols / 4;
        float local_sum = 0.0f;
        for (int c = (int)simd_lane; c < cols_vec; c += 32) {
            float4 xv = x_vec[c];
            ushort4 wv = w_vec[c];
            local_sum += xv.x * mu_bf16_to_f32(wv.x) +
                         xv.y * mu_bf16_to_f32(wv.y) +
                         xv.z * mu_bf16_to_f32(wv.z) +
                         xv.w * mu_bf16_to_f32(wv.w);
        }
        float total_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            float bias_val = mu_bf16_to_f32(bias[row]);
            out[row] = total_sum + bias_val;
        }
    } else {
        float local_sum = 0.0f;
        for (int c = (int)simd_lane; c < cols; c += 32) {
            local_sum += x[c] * mu_bf16_to_f32(w[(size_t)row * (size_t)cols + c]);
        }
        float total_sum = simd_sum(local_sum);
        if (simd_lane == 0) {
            float bias_val = mu_bf16_to_f32(bias[row]);
            out[row] = total_sum + bias_val;
        }
    }
}

kernel void mu_dense_bf16_bias_rows_simdgroup(device const float *x [[buffer(0)]],
                                              device const ushort *w [[buffer(1)]],
                                              device const ushort *bias [[buffer(2)]],
                                              device float *out [[buffer(3)]],
                                              constant int &cols [[buffer(4)]],
                                              constant int &out_cols [[buffer(5)]],
                                              constant int &x_rows [[buffer(6)]],
                                              uint2 tg [[threadgroup_position_in_grid]],
                                              uint lane [[thread_index_in_simdgroup]]) {
    int row_start = (int)tg.y * 8;
    int col_start = (int)tg.x * 32;

    threadgroup float shared_x[8 * 8];
    threadgroup float shared_w[32 * 8];

    simdgroup_matrix<float, 8, 8> acc[4];
    for (int b = 0; b < 4; b++) {
        acc[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int k = 0; k < cols; k += 8) {
        {
            int idx0 = (int)lane * 2;
            int idx1 = idx0 + 1;

            int r0 = row_start + idx0 / 8;
            int c0 = k + idx0 % 8;
            float val0 = 0.0f;
            if (r0 < x_rows && c0 < cols) {
                val0 = x[(size_t)r0 * (size_t)cols + c0];
            }
            shared_x[idx0] = val0;

            int r1 = row_start + idx1 / 8;
            int c1 = k + idx1 % 8;
            float val1 = 0.0f;
            if (r1 < x_rows && c1 < cols) {
                val1 = x[(size_t)r1 * (size_t)cols + c1];
            }
            shared_x[idx1] = val1;
        }

        {
            int oc = col_start + (int)lane;
            if (oc < out_cols && k < cols) {
                device const ushort4 *w_vec = (device const ushort4 *)(w + (size_t)oc * (size_t)cols + k);
                ushort4 v0 = w_vec[0];
                ushort4 v1 = w_vec[1];
                shared_w[lane * 8 + 0] = mu_bf16_to_f32(v0.x);
                shared_w[lane * 8 + 1] = mu_bf16_to_f32(v0.y);
                shared_w[lane * 8 + 2] = mu_bf16_to_f32(v0.z);
                shared_w[lane * 8 + 3] = mu_bf16_to_f32(v0.w);
                shared_w[lane * 8 + 4] = mu_bf16_to_f32(v1.x);
                shared_w[lane * 8 + 5] = mu_bf16_to_f32(v1.y);
                shared_w[lane * 8 + 6] = mu_bf16_to_f32(v1.z);
                shared_w[lane * 8 + 7] = mu_bf16_to_f32(v1.w);
            } else {
                for (int c_idx = 0; c_idx < 8; c_idx++) {
                    shared_w[lane * 8 + c_idx] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> x_matrix;
        simdgroup_load(x_matrix, shared_x, 8, ulong2(0, 0), false);

        for (int b = 0; b < 4; b++) {
            simdgroup_matrix<float, 8, 8> w_matrix;
            simdgroup_load(w_matrix, shared_w + b * 64, 8, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[b], x_matrix, w_matrix, acc[b]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_out[8 * 32];
    for (int b = 0; b < 4; b++) {
        simdgroup_store(acc[b], shared_out + b * 8, 32, ulong2(0, 0), false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int step = 0; step < 8; step++) {
        int linear_idx = step * 32 + (int)lane;
        int r_offset = linear_idx / 32;
        int c_offset = linear_idx % 32;

        int r = row_start + r_offset;
        int c = col_start + c_offset;

        if (r < x_rows && c < out_cols) {
            float val = shared_out[r_offset * 32 + c_offset];
            float b_val = mu_bf16_to_f32(bias[c]);
            out[(size_t)r * (size_t)out_cols + c] = mu_round_bf16(val + b_val);
        }
    }
}

kernel void mu_dense_bf16_bias_rows_simdgroup_quick_gelu(device const float *x [[buffer(0)]],
                                                        device const ushort *w [[buffer(1)]],
                                                        device const ushort *bias [[buffer(2)]],
                                                        device float *out [[buffer(3)]],
                                                        constant int &cols [[buffer(4)]],
                                                        constant int &out_cols [[buffer(5)]],
                                                        constant int &x_rows [[buffer(6)]],
                                                        uint2 tg [[threadgroup_position_in_grid]],
                                                        uint lane [[thread_index_in_simdgroup]]) {
    int row_start = (int)tg.y * 8;
    int col_start = (int)tg.x * 32;

    threadgroup float shared_x[8 * 8];
    threadgroup float shared_w[32 * 8];

    simdgroup_matrix<float, 8, 8> acc[4];
    for (int b = 0; b < 4; b++) {
        acc[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int k = 0; k < cols; k += 8) {
        {
            int idx0 = (int)lane * 2;
            int idx1 = idx0 + 1;

            int r0 = row_start + idx0 / 8;
            int c0 = k + idx0 % 8;
            float val0 = 0.0f;
            if (r0 < x_rows && c0 < cols) {
                val0 = x[(size_t)r0 * (size_t)cols + c0];
            }
            shared_x[idx0] = val0;

            int r1 = row_start + idx1 / 8;
            int c1 = k + idx1 % 8;
            float val1 = 0.0f;
            if (r1 < x_rows && c1 < cols) {
                val1 = x[(size_t)r1 * (size_t)cols + c1];
            }
            shared_x[idx1] = val1;
        }

        {
            int oc = col_start + (int)lane;
            if (oc < out_cols && k < cols) {
                device const ushort4 *w_vec = (device const ushort4 *)(w + (size_t)oc * (size_t)cols + k);
                ushort4 v0 = w_vec[0];
                ushort4 v1 = w_vec[1];
                shared_w[lane * 8 + 0] = mu_bf16_to_f32(v0.x);
                shared_w[lane * 8 + 1] = mu_bf16_to_f32(v0.y);
                shared_w[lane * 8 + 2] = mu_bf16_to_f32(v0.z);
                shared_w[lane * 8 + 3] = mu_bf16_to_f32(v0.w);
                shared_w[lane * 8 + 4] = mu_bf16_to_f32(v1.x);
                shared_w[lane * 8 + 5] = mu_bf16_to_f32(v1.y);
                shared_w[lane * 8 + 6] = mu_bf16_to_f32(v1.z);
                shared_w[lane * 8 + 7] = mu_bf16_to_f32(v1.w);
            } else {
                for (int c_idx = 0; c_idx < 8; c_idx++) {
                    shared_w[lane * 8 + c_idx] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> x_matrix;
        simdgroup_load(x_matrix, shared_x, 8, ulong2(0, 0), false);

        for (int b = 0; b < 4; b++) {
            simdgroup_matrix<float, 8, 8> w_matrix;
            simdgroup_load(w_matrix, shared_w + b * 64, 8, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[b], x_matrix, w_matrix, acc[b]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_out[8 * 32];
    for (int b = 0; b < 4; b++) {
        simdgroup_store(acc[b], shared_out + b * 8, 32, ulong2(0, 0), false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int step = 0; step < 8; step++) {
        int linear_idx = step * 32 + (int)lane;
        int r_offset = linear_idx / 32;
        int c_offset = linear_idx % 32;

        int r = row_start + r_offset;
        int c = col_start + c_offset;

        if (r < x_rows && c < out_cols) {
            float val = shared_out[r_offset * 32 + c_offset];
            float b_val = mu_bf16_to_f32(bias[c]);
            float sum = val + b_val;
            float act = sum / (1.0f + exp(-1.702f * sum));
            out[(size_t)r * (size_t)out_cols + c] = mu_round_bf16(act);
        }
    }
}

static inline float mu_erf_approx(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) +
                    1.421413741f) * t - 0.284496736f) * t +
                  0.254829592f) * t;
    return sign * (1.0f - poly * exp(-ax * ax));
}

kernel void mu_dense_bf16_bias_rows_simdgroup_gelu(device const float *x [[buffer(0)]],
                                                  device const ushort *w [[buffer(1)]],
                                                  device const ushort *bias [[buffer(2)]],
                                                  device float *out [[buffer(3)]],
                                                  constant int &cols [[buffer(4)]],
                                                  constant int &out_cols [[buffer(5)]],
                                                  constant int &x_rows [[buffer(6)]],
                                                  uint2 tg [[threadgroup_position_in_grid]],
                                                  uint lane [[thread_index_in_simdgroup]]) {
    int row_start = (int)tg.y * 8;
    int col_start = (int)tg.x * 32;

    threadgroup float shared_x[8 * 8];
    threadgroup float shared_w[32 * 8];

    simdgroup_matrix<float, 8, 8> acc[4];
    for (int b = 0; b < 4; b++) {
        acc[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int k = 0; k < cols; k += 8) {
        {
            int idx0 = (int)lane * 2;
            int idx1 = idx0 + 1;

            int r0 = row_start + idx0 / 8;
            int c0 = k + idx0 % 8;
            float val0 = 0.0f;
            if (r0 < x_rows && c0 < cols) {
                val0 = x[(size_t)r0 * (size_t)cols + c0];
            }
            shared_x[idx0] = val0;

            int r1 = row_start + idx1 / 8;
            int c1 = k + idx1 % 8;
            float val1 = 0.0f;
            if (r1 < x_rows && c1 < cols) {
                val1 = x[(size_t)r1 * (size_t)cols + c1];
            }
            shared_x[idx1] = val1;
        }

        {
            int oc = col_start + (int)lane;
            if (oc < out_cols && k < cols) {
                device const ushort4 *w_vec = (device const ushort4 *)(w + (size_t)oc * (size_t)cols + k);
                ushort4 v0 = w_vec[0];
                ushort4 v1 = w_vec[1];
                shared_w[lane * 8 + 0] = mu_bf16_to_f32(v0.x);
                shared_w[lane * 8 + 1] = mu_bf16_to_f32(v0.y);
                shared_w[lane * 8 + 2] = mu_bf16_to_f32(v0.z);
                shared_w[lane * 8 + 3] = mu_bf16_to_f32(v0.w);
                shared_w[lane * 8 + 4] = mu_bf16_to_f32(v1.x);
                shared_w[lane * 8 + 5] = mu_bf16_to_f32(v1.y);
                shared_w[lane * 8 + 6] = mu_bf16_to_f32(v1.z);
                shared_w[lane * 8 + 7] = mu_bf16_to_f32(v1.w);
            } else {
                for (int c_idx = 0; c_idx < 8; c_idx++) {
                    shared_w[lane * 8 + c_idx] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> x_matrix;
        simdgroup_load(x_matrix, shared_x, 8, ulong2(0, 0), false);

        for (int b = 0; b < 4; b++) {
            simdgroup_matrix<float, 8, 8> w_matrix;
            simdgroup_load(w_matrix, shared_w + b * 64, 8, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[b], x_matrix, w_matrix, acc[b]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_out[8 * 32];
    for (int b = 0; b < 4; b++) {
        simdgroup_store(acc[b], shared_out + b * 8, 32, ulong2(0, 0), false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int step = 0; step < 8; step++) {
        int linear_idx = step * 32 + (int)lane;
        int r_offset = linear_idx / 32;
        int c_offset = linear_idx % 32;

        int r = row_start + r_offset;
        int c = col_start + c_offset;

        if (r < x_rows && c < out_cols) {
            float val = shared_out[r_offset * 32 + c_offset];
            float b_val = mu_bf16_to_f32(bias[c]);
            float sum = val + b_val;
            float act = 0.5f * sum * (1.0f + mu_erf_approx(sum * 0.7071067811865476f));
            out[(size_t)r * (size_t)out_cols + c] = mu_round_bf16(act);
        }
    }
}


