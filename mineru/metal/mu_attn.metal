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

static inline float mu_text_rope_value_pos(device const float *head,
                                           device const int *position_ids,
                                           int seq,
                                           int token_index,
                                           int d) {
    int axis = mu_text_rope_axis(d);
    int pos = position_ids[(size_t)axis * (size_t)seq + (size_t)token_index];
    int inv_idx = d < 32 ? d : d - 32;
    float inv = pow(1000000.0f, -((float)(2 * inv_idx) / 64.0f));
    float angle = (float)pos * inv;
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

kernel void mu_text_attn_seq_pos(device const float *q [[buffer(0)]],
                                 device const float *k [[buffer(1)]],
                                 device const float *v [[buffer(2)]],
                                 device const int *position_ids [[buffer(3)]],
                                 device float *out [[buffer(4)]],
                                 constant int &seq [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int kv_group = 7;
    const int head_dim = 64;
    int t = (int)gid.x;
    int head = (int)gid.y;
    if (t >= seq || head >= n_heads) return;

    int kvh = head / kv_group;
    device const float *q_head = q + ((size_t)t * n_heads + head) * head_dim;

    float qd[64];
    for (int d = 0; d < head_dim; d++) {
        qd[d] = mu_text_rope_value_pos(q_head, position_ids, seq, t, d);
    }

    float max_score = -3.402823466e38f;
    for (int sidx = 0; sidx <= t; sidx++) {
        device const float *k_head = k + ((size_t)sidx * n_kv_heads + kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_text_rope_value_pos(k_head, position_ids, seq, sidx, d);
        }
        float score = dot * 0.125f;
        if (score > max_score) max_score = score;
    }

    float denom = 0.0f;
    for (int chunk = 0; chunk <= t; chunk += 64) {
        int n = min(64, t - chunk + 1);
        for (int i = 0; i < n; i++) {
            int sidx = chunk + i;
            device const float *k_head = k + ((size_t)sidx * n_kv_heads + kvh) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += qd[d] * mu_text_rope_value_pos(k_head, position_ids, seq, sidx, d);
            }
            denom += exp(dot * 0.125f - max_score);
        }
    }

    device float *oh = out + ((size_t)t * n_heads + head) * head_dim;
    float acc[64];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    for (int sidx = 0; sidx <= t; sidx++) {
        device const float *k_head = k + ((size_t)sidx * n_kv_heads + kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_text_rope_value_pos(k_head, position_ids, seq, sidx, d);
        }
        float p = exp(dot * 0.125f - max_score) / denom;
        device const float *v_head = v + ((size_t)sidx * n_kv_heads + kvh) * head_dim;
        for (int d = 0; d < head_dim; d++) acc[d] += p * v_head[d];
    }
    for (int d = 0; d < head_dim; d++) oh[d] = acc[d];
}

kernel void mu_text_attn_cached(device const float *q [[buffer(0)]],
                                device const float *k_cache [[buffer(1)]],
                                device const float *v_cache [[buffer(2)]],
                                device float *out [[buffer(3)]],
                                constant int &cache_len [[buffer(4)]],
                                uint head_gid [[thread_position_in_grid]]) {
    const int n_heads = 14;
    const int kv_group = 7;
    const int head_dim = 64;
    if (head_gid >= (uint)n_heads || cache_len <= 0) return;

    int head = (int)head_gid;
    int kvh = head / kv_group;
    device const float *q_head = q + (size_t)head * head_dim;

    float max_score = -3.402823466e38f;
    for (int sidx = 0; sidx < cache_len; sidx++) {
        device const float *k_head =
            k_cache + ((size_t)sidx * 2u + (size_t)kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q_head[d] * k_head[d];
        float score = dot * 0.125f;
        if (score > max_score) max_score = score;
    }

    float denom = 0.0f;
    for (int sidx = 0; sidx < cache_len; sidx++) {
        device const float *k_head =
            k_cache + ((size_t)sidx * 2u + (size_t)kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q_head[d] * k_head[d];
        denom += exp(dot * 0.125f - max_score);
    }

    device float *oh = out + (size_t)head * head_dim;
    float acc[64];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    for (int sidx = 0; sidx < cache_len; sidx++) {
        device const float *k_head =
            k_cache + ((size_t)sidx * 2u + (size_t)kvh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q_head[d] * k_head[d];
        float p = exp(dot * 0.125f - max_score) / denom;
        device const float *v_head =
            v_cache + ((size_t)sidx * 2u + (size_t)kvh) * head_dim;
        for (int d = 0; d < head_dim; d++) acc[d] += p * v_head[d];
    }
    for (int d = 0; d < head_dim; d++) oh[d] = acc[d];
}

kernel void mu_text_rope_cache_update(device float *q [[buffer(0)]],
                                      device float *k [[buffer(1)]],
                                      device const float *v [[buffer(2)]],
                                      device float *k_cache [[buffer(3)]],
                                      device float *v_cache [[buffer(4)]],
                                      constant int *pos3 [[buffer(5)]],
                                      constant int &cache_pos [[buffer(6)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (gid < 448) {
        int head = (int)gid / 32;
        int d = (int)gid - head * 32;
        int axis = mu_text_rope_axis(d);
        float inv = pow(1000000.0f, -((float)(2 * d) / 64.0f));
        float angle = (float)pos3[axis] * inv;
        float c = cos(angle);
        float s = sin(angle);
        device float *head_q = q + (size_t)head * 64u;
        float lo = head_q[d];
        float hi = head_q[d + 32];
        head_q[d] = lo * c - hi * s;
        head_q[d + 32] = hi * c + lo * s;
    }

    if (gid < 64) {
        int head = (int)gid / 32;
        int d = (int)gid - head * 32;
        int axis = mu_text_rope_axis(d);
        float inv = pow(1000000.0f, -((float)(2 * d) / 64.0f));
        float angle = (float)pos3[axis] * inv;
        float c = cos(angle);
        float s = sin(angle);
        device float *head_k = k + (size_t)head * 64u;
        float lo = head_k[d];
        float hi = head_k[d + 32];
        size_t base = (size_t)cache_pos * 128u + (size_t)head * 64u;
        float rot_lo = lo * c - hi * s;
        float rot_hi = hi * c + lo * s;
        head_k[d] = rot_lo;
        head_k[d + 32] = rot_hi;
        k_cache[base + (size_t)d] = rot_lo;
        k_cache[base + (size_t)d + 32u] = rot_hi;
    }

    if (gid < 128) {
        v_cache[(size_t)cache_pos * 128u + (size_t)gid] = v[gid];
    }
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

kernel void mu_text_attn_cached_simd(device const float *q [[buffer(0)]],
                                     device const float *k_cache [[buffer(1)]],
                                     device const float *v_cache [[buffer(2)]],
                                     device float *out [[buffer(3)]],
                                     constant int &cache_len [[buffer(4)]],
                                     uint2 gid [[thread_position_in_grid]],
                                     uint simd_lane [[thread_index_in_simdgroup]]) {
    int head = gid.x / 32;
    if (head >= 14 || cache_len <= 0) return;
    int kvh = head / 7;
    int tid = (int)simd_lane;

    device const float *q_head = q + (size_t)head * 64u;
    float q0 = q_head[tid];
    float q1 = q_head[tid + 32];

    float max_score = -3.402823466e38f;
    threadgroup float scores[4096];

    for (int sidx = 0; sidx < cache_len; sidx++) {
        device const float *k_head = k_cache + ((size_t)sidx * 2u + (size_t)kvh) * 64u;
        float pdot = q0 * k_head[tid] + q1 * k_head[tid + 32];
        float dot = simd_sum(pdot);
        float score = dot * 0.125f;
        if (simd_lane == 0) {
            scores[sidx] = score;
        }
        if (score > max_score) max_score = score;
    }
    max_score = simd_max(max_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    for (int sidx = tid; sidx < cache_len; sidx += 32) {
        local_denom += exp(scores[sidx] - max_score);
    }
    float denom = simd_sum(local_denom);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int sidx = 0; sidx < cache_len; sidx++) {
        float p = exp(scores[sidx] - max_score) / denom;
        device const float *v_head = v_cache + ((size_t)sidx * 2u + (size_t)kvh) * 64u;
        acc0 += p * v_head[tid];
        acc1 += p * v_head[tid + 32];
    }

    device float *oh = out + (size_t)head * 64u;
    oh[tid] = acc0;
    oh[tid + 32] = acc1;
}
