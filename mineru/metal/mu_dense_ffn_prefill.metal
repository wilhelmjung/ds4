#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

static inline float mu_bf16_to_f32(ushort v) {
    uint bits = ((uint)v) << 16;
    return as_type<float>(bits);
}

kernel void mu_dense_bf16_rows_simdgroup_swiglu(device const float *x [[buffer(0)]],
                                                device const ushort *w_gate [[buffer(1)]],
                                                device const ushort *w_up [[buffer(2)]],
                                                device float *out [[buffer(3)]],
                                                constant int &cols [[buffer(4)]],
                                                constant int &out_cols [[buffer(5)]],
                                                constant int &x_rows [[buffer(6)]],
                                                uint2 tg [[threadgroup_position_in_grid]],
                                                uint lane [[thread_index_in_simdgroup]]) {
    int row_start = (int)tg.y * 8;
    int col_start = (int)tg.x * 32;

    threadgroup float shared_x[8 * 8];
    threadgroup float shared_w_gate[32 * 8];
    threadgroup float shared_w_up[32 * 8];

    simdgroup_matrix<float, 8, 8> acc_gate[4];
    simdgroup_matrix<float, 8, 8> acc_up[4];
    for (int b = 0; b < 4; b++) {
        acc_gate[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc_up[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (int k = 0; k < cols; k += 8) {
        // Load x to shared memory
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

        // Load w_gate to shared memory
        {
            int oc = col_start + (int)lane;
            if (oc < out_cols && k < cols) {
                device const ushort4 *w_vec = (device const ushort4 *)(w_gate + (size_t)oc * (size_t)cols + k);
                ushort4 v0 = w_vec[0];
                ushort4 v1 = w_vec[1];
                shared_w_gate[lane * 8 + 0] = mu_bf16_to_f32(v0.x);
                shared_w_gate[lane * 8 + 1] = mu_bf16_to_f32(v0.y);
                shared_w_gate[lane * 8 + 2] = mu_bf16_to_f32(v0.z);
                shared_w_gate[lane * 8 + 3] = mu_bf16_to_f32(v0.w);
                shared_w_gate[lane * 8 + 4] = mu_bf16_to_f32(v1.x);
                shared_w_gate[lane * 8 + 5] = mu_bf16_to_f32(v1.y);
                shared_w_gate[lane * 8 + 6] = mu_bf16_to_f32(v1.z);
                shared_w_gate[lane * 8 + 7] = mu_bf16_to_f32(v1.w);
            } else {
                for (int c_idx = 0; c_idx < 8; c_idx++) {
                    shared_w_gate[lane * 8 + c_idx] = 0.0f;
                }
            }
        }

        // Load w_up to shared memory
        {
            int oc = col_start + (int)lane;
            if (oc < out_cols && k < cols) {
                device const ushort4 *w_vec = (device const ushort4 *)(w_up + (size_t)oc * (size_t)cols + k);
                ushort4 v0 = w_vec[0];
                ushort4 v1 = w_vec[1];
                shared_w_up[lane * 8 + 0] = mu_bf16_to_f32(v0.x);
                shared_w_up[lane * 8 + 1] = mu_bf16_to_f32(v0.y);
                shared_w_up[lane * 8 + 2] = mu_bf16_to_f32(v0.z);
                shared_w_up[lane * 8 + 3] = mu_bf16_to_f32(v0.w);
                shared_w_up[lane * 8 + 4] = mu_bf16_to_f32(v1.x);
                shared_w_up[lane * 8 + 5] = mu_bf16_to_f32(v1.y);
                shared_w_up[lane * 8 + 6] = mu_bf16_to_f32(v1.z);
                shared_w_up[lane * 8 + 7] = mu_bf16_to_f32(v1.w);
            } else {
                for (int c_idx = 0; c_idx < 8; c_idx++) {
                    shared_w_up[lane * 8 + c_idx] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> x_matrix;
        simdgroup_load(x_matrix, shared_x, 8, ulong2(0, 0), false);

        for (int b = 0; b < 4; b++) {
            simdgroup_matrix<float, 8, 8> w_gate_matrix;
            simdgroup_load(w_gate_matrix, shared_w_gate + b * 64, 8, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc_gate[b], x_matrix, w_gate_matrix, acc_gate[b]);

            simdgroup_matrix<float, 8, 8> w_up_matrix;
            simdgroup_load(w_up_matrix, shared_w_up + b * 64, 8, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc_up[b], x_matrix, w_up_matrix, acc_up[b]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_gate_out[8 * 32];
    threadgroup float shared_up_out[8 * 32];
    for (int b = 0; b < 4; b++) {
        simdgroup_store(acc_gate[b], shared_gate_out + b * 8, 32, ulong2(0, 0), false);
        simdgroup_store(acc_up[b], shared_up_out + b * 8, 32, ulong2(0, 0), false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int step = 0; step < 8; step++) {
        int linear_idx = step * 32 + (int)lane;
        int r_offset = linear_idx / 32;
        int c_offset = linear_idx % 32;

        int r = row_start + r_offset;
        int c = col_start + c_offset;

        if (r < x_rows && c < out_cols) {
            float g = shared_gate_out[r_offset * 32 + c_offset];
            float u = shared_up_out[r_offset * 32 + c_offset];
            float silu_g = g / (1.0f + exp(-g));
            float val = silu_g * u;
            out[(size_t)r * (size_t)out_cols + c] = val;
        }
    }
}
