#include <metal_stdlib>
using namespace metal;

kernel void mu_text_attn_token0(device const float *v [[buffer(0)]],
                                device float *out [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
    const int head_dim = 64;
    const int kv_group = 7;
    const int hidden = 896;
    if (gid >= hidden) return;
    int head = (int)gid / head_dim;
    int dim = (int)gid - head * head_dim;
    int kvh = head / kv_group;
    out[gid] = v[kvh * head_dim + dim];
}

kernel void mu_add_f32(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       constant int &n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= n) return;
    out[gid] = a[gid] + b[gid];
}
