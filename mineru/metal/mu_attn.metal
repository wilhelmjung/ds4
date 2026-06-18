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

static inline int mu_text_rope_axis(int d) {
    if (d < 8) return 0;
    if (d < 20) return 1;
    if (d < 32) return 2;
    if (d < 40) return 0;
    if (d < 52) return 1;
    return 2;
}

static inline float mu_text_rope_value(device const float *head,
                                       int token_index,
                                       int d) {
    int inv_idx = d < 32 ? d : d - 32;
    float inv = pow(1000000.0f, -((float)(2 * inv_idx) / 64.0f));
    float angle = (float)token_index * inv;
    float c = cos(angle);
    float s = sin(angle);
    float old = head[d];
    float rot = d < 32 ? -head[d + 32] : head[d - 32];
    return old * c + rot * s;
}

kernel void mu_text_attn_seq(device const float *q [[buffer(0)]],
                             device const float *k [[buffer(1)]],
                             device const float *v [[buffer(2)]],
                             device float *out [[buffer(3)]],
                             constant int &seq [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int kv_group = 7;
    const int head_dim = 64;
    const int hidden = 896;
    if (gid >= (uint)(seq * hidden)) return;

    int t = (int)gid / hidden;
    int hd = (int)gid - t * hidden;
    int head = hd / head_dim;
    int dim = hd - head * head_dim;
    int kvh = head / kv_group;
    device const float *q_head = q + ((size_t)t * n_heads + head) * head_dim;

    float qd[64];
    for (int d = 0; d < head_dim; d++) {
        qd[d] = mu_text_rope_value(q_head, t, d);
    }

    float max_score = -3.402823466e38f;
    for (int sidx = 0; sidx <= t; sidx++) {
        device const float *k_head = k + ((size_t)sidx * n_kv_heads + kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_text_rope_value(k_head, sidx, d);
        }
        float score = dot * 0.125f;
        if (score > max_score) max_score = score;
    }

    float denom = 0.0f;
    for (int sidx = 0; sidx <= t; sidx++) {
        device const float *k_head = k + ((size_t)sidx * n_kv_heads + kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_text_rope_value(k_head, sidx, d);
        }
        denom += exp(dot * 0.125f - max_score);
    }

    float acc = 0.0f;
    for (int sidx = 0; sidx <= t; sidx++) {
        device const float *k_head = k + ((size_t)sidx * n_kv_heads + kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_text_rope_value(k_head, sidx, d);
        }
        float p = exp(dot * 0.125f - max_score) / denom;
        device const float *v_head = v + ((size_t)sidx * n_kv_heads + kvh) * head_dim;
        acc += p * v_head[dim];
    }
    out[gid] = acc;
}

kernel void mu_add_f32(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       constant int &n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= n) return;
    out[gid] = a[gid] + b[gid];
}

kernel void mu_silu_mul_f32(device const float *gate [[buffer(0)]],
                            device const float *up [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            constant int &n [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= n) return;
    float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}
