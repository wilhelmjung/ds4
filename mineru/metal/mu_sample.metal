#include <metal_stdlib>
using namespace metal;

static inline float f32_to_bf16_to_f32(float f) {
    union {
        float fval;
        uint32_t uval;
    } u;
    u.fval = f;
    uint32_t lsb = (u.uval >> 16) & 1u;
    u.uval += 0x7fffu + lsb;
    u.uval &= 0xffff0000u;
    return u.fval;
}

kernel void mu_argmax_f32(device const float *logits [[buffer(0)]],
                          device int *out_id [[buffer(1)]],
                          device float *out_val [[buffer(2)]],
                          constant int &n [[buffer(3)]],
                          uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float local_max[512];
    threadgroup int local_idx[512];

    float best_val = -3.402823466e38f;
    int best_idx = -1;

    for (int i = (int)tid; i < n; i += 512) {
        float val = f32_to_bf16_to_f32(logits[i]);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    local_max[tid] = best_val;
    local_idx[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 256; stride > 0; stride /= 2) {
        if (tid < stride) {
            float other_val = local_max[tid + stride];
            int other_idx = local_idx[tid + stride];
            if (other_val > local_max[tid]) {
                local_max[tid] = other_val;
                local_idx[tid] = other_idx;
            } else if (other_val == local_max[tid]) {
                if (other_idx >= 0 && (local_idx[tid] < 0 || other_idx < local_idx[tid])) {
                    local_idx[tid] = other_idx;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        *out_id = local_idx[0];
        *out_val = local_max[0];
    }
}
