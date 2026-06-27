#include <metal_stdlib>
using namespace metal;

static inline float mu_bf16_to_f32(ushort val) {
    uint temp = (uint)val << 16;
    return as_type<float>(temp);
}

kernel void mu_vision_fused_ffn(device const float *hs_in [[buffer(0)]],
                                device const ushort *fc1_w [[buffer(1)]],
                                device const ushort *fc1_b [[buffer(2)]],
                                device const ushort *fc2_w [[buffer(3)]],
                                device const ushort *fc2_b [[buffer(4)]],
                                device float *out [[buffer(5)]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    
    threadgroup float shared_hs[1280];
    threadgroup float shared_mid[5120];

    // 1. Cooperatively load the input row of size 1280 into threadgroup memory
    device const float4 *hs_in_vec = (device const float4 *)(hs_in + row * 1280u);
    threadgroup float4 *shared_hs_vec = (threadgroup float4 *)shared_hs;
    for (int i = (int)tid; i < 320; i += 128) {
        shared_hs_vec[i] = hs_in_vec[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Cooperatively compute the fc1 projection + QuickGELU activation
    // fc1_w shape is [5120, 1280]
    for (int c = (int)tid; c < 5120; c += 128) {
        float sum = 0.0f;
        device const ushort4 *w_vec = (device const ushort4 *)(fc1_w + (size_t)c * 1280u);
        threadgroup const float4 *hs_vec = (threadgroup const float4 *)shared_hs;
        for (int k = 0; k < 320; k++) {
            ushort4 w4 = w_vec[k];
            float4 h4 = hs_vec[k];
            sum += h4.x * mu_bf16_to_f32(w4.x) +
                   h4.y * mu_bf16_to_f32(w4.y) +
                   h4.z * mu_bf16_to_f32(w4.z) +
                   h4.w * mu_bf16_to_f32(w4.w);
        }
        sum += mu_bf16_to_f32(fc1_b[c]);
        // QuickGELU: x * sigmoid(1.702 * x) = x / (1.0f + exp(-1.702f * x))
        shared_mid[c] = sum / (1.0f + exp(-1.702f * sum));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Cooperatively compute the fc2 down-projection
    // fc2_w shape is [1280, 5120]
    for (int c = (int)tid; c < 1280; c += 128) {
        float sum = 0.0f;
        device const ushort4 *w_vec = (device const ushort4 *)(fc2_w + (size_t)c * 5120u);
        threadgroup const float4 *mid_vec = (threadgroup const float4 *)shared_mid;
        for (int k = 0; k < 1280; k++) {
            ushort4 w4 = w_vec[k];
            float4 m4 = mid_vec[k];
            sum += m4.x * mu_bf16_to_f32(w4.x) +
                   m4.y * mu_bf16_to_f32(w4.y) +
                   m4.z * mu_bf16_to_f32(w4.z) +
                   m4.w * mu_bf16_to_f32(w4.w);
        }
        sum += mu_bf16_to_f32(fc2_b[c]);
        out[row * 1280u + c] = sum;
    }
}
