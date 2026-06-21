#include <metal_stdlib>
using namespace metal;

static inline float mu_bf16_to_f32(ushort val) {
    uint temp = (uint)val << 16;
    return as_type<float>(temp);
}

kernel void mu_text_decode_fused_ffn(device const float *hs_in [[buffer(0)]],
                                     device const ushort *post_norm_w [[buffer(1)]],
                                     device const ushort *gate_w [[buffer(2)]],
                                     device const ushort *up_w [[buffer(3)]],
                                     device const ushort *down_w [[buffer(4)]],
                                     device float *out [[buffer(5)]],
                                     constant float &eps [[buffer(6)]],
                                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared_hs[896];
    threadgroup float shared_normed[896];
    threadgroup float shared_sums[256];
    threadgroup float shared_mid[4864];

    // 1. Load input hidden state into threadgroup memory
    for (int i = (int)tid; i < 896; i += 256) {
        shared_hs[i] = hs_in[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Compute local sum of squares for RMSNorm
    float local_sum = 0.0f;
    for (int i = (int)tid; i < 896; i += 256) {
        local_sum += shared_hs[i] * shared_hs[i];
    }
    shared_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum of squares cooperatively
    if (tid < 128) { shared_sums[tid] += shared_sums[tid + 128]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) { shared_sums[tid] += shared_sums[tid + 64]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float s = shared_sums[tid] + shared_sums[tid + 32];
        s = simd_sum(s);
        if (tid == 0) shared_sums[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = rsqrt(shared_sums[0] / 896.0f + eps);
    for (int i = (int)tid; i < 896; i += 256) {
        shared_normed[i] = shared_hs[i] * rms * mu_bf16_to_f32(post_norm_w[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Compute Gate and Up projections + SiLU multiplication
    for (int row = (int)tid; row < 4864; row += 256) {
        float sum_gate = 0.0f;
        float sum_up = 0.0f;
        device const ushort *gate_row = gate_w + (size_t)row * 896u;
        device const ushort *up_row = up_w + (size_t)row * 896u;
        
        device const ushort4 *gate_row_vec = (device const ushort4 *)gate_row;
        device const ushort4 *up_row_vec = (device const ushort4 *)up_row;
        for (int c_vec = 0; c_vec < 224; c_vec++) {
            ushort4 g = gate_row_vec[c_vec];
            ushort4 u = up_row_vec[c_vec];
            int c = c_vec * 4;
            float n0 = shared_normed[c + 0];
            float n1 = shared_normed[c + 1];
            float n2 = shared_normed[c + 2];
            float n3 = shared_normed[c + 3];
            sum_gate += n0 * mu_bf16_to_f32(g.x) +
                        n1 * mu_bf16_to_f32(g.y) +
                        n2 * mu_bf16_to_f32(g.z) +
                        n3 * mu_bf16_to_f32(g.w);
            sum_up   += n0 * mu_bf16_to_f32(u.x) +
                        n1 * mu_bf16_to_f32(u.y) +
                        n2 * mu_bf16_to_f32(u.z) +
                        n3 * mu_bf16_to_f32(u.w);
        }
        
        float gate_val = sum_gate;
        float silu_gate = gate_val / (1.0f + exp(-gate_val));
        shared_mid[row] = silu_gate * sum_up;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Compute Down projection and perform residual addition
    for (int row = (int)tid; row < 896; row += 256) {
        float sum_down = 0.0f;
        device const ushort *down_row = down_w + (size_t)row * 4864u;
        device const ushort4 *down_row_vec = (device const ushort4 *)down_row;
        for (int c_vec = 0; c_vec < 1216; c_vec++) {
            ushort4 d = down_row_vec[c_vec];
            int c = c_vec * 4;
            sum_down += shared_mid[c + 0] * mu_bf16_to_f32(d.x) +
                        shared_mid[c + 1] * mu_bf16_to_f32(d.y) +
                        shared_mid[c + 2] * mu_bf16_to_f32(d.z) +
                        shared_mid[c + 3] * mu_bf16_to_f32(d.w);
        }
        out[row] = shared_hs[row] + sum_down;
    }
}
