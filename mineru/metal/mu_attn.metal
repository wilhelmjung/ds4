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

kernel void mu_text_prefill_rope_cache_update(device const float *k [[buffer(0)]],
                                              device const float *v [[buffer(1)]],
                                              device const int *position_ids [[buffer(2)]],
                                              device float *k_cache [[buffer(3)]],
                                              device float *v_cache [[buffer(4)]],
                                              constant int &seq [[buffer(5)]],
                                              constant int &cache_cap [[buffer(6)]],
                                              constant int &layer [[buffer(7)]],
                                              uint2 gid [[thread_position_in_grid]]) {
    int t = (int)gid.x;
    int head = (int)gid.y;
    if (t >= seq || head >= 2) return;

    const int head_dim = 64;
    size_t base_k = ((size_t)t * 2u + (size_t)head) * head_dim;
    size_t base_cache = (((size_t)layer * (size_t)cache_cap) + (size_t)t) * 128u + (size_t)head * 64u;

    for (int d = 0; d < 32; d++) {
        int axis = mu_text_rope_axis(d);
        int pos = position_ids[(size_t)axis * (size_t)seq + (size_t)t];
        float inv = pow(1000000.0f, -((float)(2 * d) / 64.0f));
        float angle = (float)pos * inv;
        float c = cos(angle);
        float s = sin(angle);

        float lo = k[base_k + d];
        float hi = k[base_k + d + 32];
        float rot_lo = lo * c - hi * s;
        float rot_hi = hi * c + lo * s;

        k_cache[base_cache + (size_t)d] = rot_lo;
        k_cache[base_cache + (size_t)d + 32u] = rot_hi;
    }

    if (head == 0) {
        size_t base_v = (size_t)t * 128u;
        size_t base_v_cache = (((size_t)layer * (size_t)cache_cap) + (size_t)t) * 128u;
        for (int d = 0; d < 128; d++) {
            v_cache[base_v_cache + d] = v[base_v + d];
        }
    }
}

kernel void mu_text_prefill_attn_flash(device const float *q [[buffer(0)]],
                                       device const float *k_cache [[buffer(1)]],
                                       device const float *v_cache [[buffer(2)]],
                                       device float *out [[buffer(3)]],
                                       constant int &seq [[buffer(4)]],
                                       constant int &cache_cap [[buffer(5)]],
                                       constant int &layer [[buffer(6)]],
                                       uint2 tg [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    int head = (int)tg.y;
    int query_row = (int)tg.x * 32 + (int)lane;
    const int head_dim = 64;
    const int n_heads = 14;
    const int kv_group = 7;
    const int kvh = head / kv_group;
    const float scale = 0.125f; // 1 / sqrt(64)

    float qd[64];
    if (query_row < seq) {
        device const float *q_head = q + ((size_t)query_row * (size_t)n_heads + (size_t)head) * head_dim;
        for (int d = 0; d < 64; d++) {
            qd[d] = mu_text_rope_value(q_head, query_row, d);
        }
    }

    threadgroup float shared_k[32 * 64];
    threadgroup float shared_v[32 * 64];

    float max_score = -3.402823466e38f;
    float denom = 0.0f;

    // Pass 1: compute online softmax stats (max_score and denom) causally
    for (int kb = 0; kb < seq; kb += 32) {
        int k_row = kb + (int)lane;
        if (k_row < seq) {
            size_t base_k_cache = (((size_t)layer * (size_t)cache_cap) + (size_t)k_row) * 128u + (size_t)kvh * 64u;
            for (int d = 0; d < 64; d++) {
                shared_k[lane * 64 + d] = k_cache[base_k_cache + d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query_row < seq && kb <= query_row) {
            int limit = min(32, query_row - kb + 1);
            for (int j = 0; j < limit; j++) {
                float dot = 0.0f;
                threadgroup const float *k_shared = shared_k + j * 64;
                for (int d = 0; d < 64; d++) {
                    dot += qd[d] * k_shared[d];
                }
                float score = dot * scale;
                float m_new = max(max_score, score);
                denom = denom * exp(max_score - m_new) + exp(score - m_new);
                max_score = m_new;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: accumulate value vectors weighted by softmax probabilities
    float acc[64];
    for (int d = 0; d < 64; d++) acc[d] = 0.0f;

    for (int kb = 0; kb < seq; kb += 32) {
        int kv_row = kb + (int)lane;
        if (kv_row < seq) {
            size_t base_k_cache = (((size_t)layer * (size_t)cache_cap) + (size_t)kv_row) * 128u + (size_t)kvh * 64u;
            size_t base_v_cache = (((size_t)layer * (size_t)cache_cap) + (size_t)kv_row) * 128u + (size_t)kvh * 64u;
            for (int d = 0; d < 64; d++) {
                shared_k[lane * 64 + d] = k_cache[base_k_cache + d];
                shared_v[lane * 64 + d] = v_cache[base_v_cache + d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query_row < seq && kb <= query_row) {
            int limit = min(32, query_row - kb + 1);
            for (int j = 0; j < limit; j++) {
                float dot = 0.0f;
                threadgroup const float *k_shared = shared_k + j * 64;
                for (int d = 0; d < 64; d++) {
                    dot += qd[d] * k_shared[d];
                }
                float p = exp(dot * scale - max_score) / denom;
                threadgroup const float *v_shared = shared_v + j * 64;
                for (int d = 0; d < 64; d++) {
                    acc[d] += p * v_shared[d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_row < seq) {
        device float *oh = out + ((size_t)query_row * (size_t)n_heads + (size_t)head) * head_dim;
        for (int d = 0; d < 64; d++) {
            oh[d] = acc[d];
        }
    }
}

kernel void mu_text_prefill_attn_pos_flash(device const float *q [[buffer(0)]],
                                           device const float *k_cache [[buffer(1)]],
                                           device const float *v_cache [[buffer(2)]],
                                           device const int *position_ids [[buffer(3)]],
                                           device float *out [[buffer(4)]],
                                           constant int &seq [[buffer(5)]],
                                           constant int &cache_cap [[buffer(6)]],
                                           constant int &layer [[buffer(7)]],
                                           uint2 tg [[threadgroup_position_in_grid]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    int head = (int)tg.y;
    int query_row = (int)tg.x * 32 + (int)lane;
    const int head_dim = 64;
    const int n_heads = 14;
    const int kv_group = 7;
    const int kvh = head / kv_group;
    const float scale = 0.125f; // 1 / sqrt(64)

    float qd[64];
    if (query_row < seq) {
        device const float *q_head = q + ((size_t)query_row * (size_t)n_heads + (size_t)head) * head_dim;
        for (int d = 0; d < 64; d++) {
            qd[d] = mu_text_rope_value_pos(q_head, position_ids, seq, query_row, d);
        }
    }

    threadgroup float shared_k[32 * 64];
    threadgroup float shared_v[32 * 64];

    float max_score = -3.402823466e38f;
    float denom = 0.0f;

    // Pass 1: compute online softmax stats (max_score and denom) causally
    for (int kb = 0; kb < seq; kb += 32) {
        int k_row = kb + (int)lane;
        if (k_row < seq) {
            size_t base_k_cache = (((size_t)layer * (size_t)cache_cap) + (size_t)k_row) * 128u + (size_t)kvh * 64u;
            for (int d = 0; d < 64; d++) {
                shared_k[lane * 64 + d] = k_cache[base_k_cache + d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query_row < seq && kb <= query_row) {
            int limit = min(32, query_row - kb + 1);
            for (int j = 0; j < limit; j++) {
                float dot = 0.0f;
                threadgroup const float *k_shared = shared_k + j * 64;
                for (int d = 0; d < 64; d++) {
                    dot += qd[d] * k_shared[d];
                }
                float score = dot * scale;
                float m_new = max(max_score, score);
                denom = denom * exp(max_score - m_new) + exp(score - m_new);
                max_score = m_new;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: accumulate value vectors weighted by softmax probabilities
    float acc[64];
    for (int d = 0; d < 64; d++) acc[d] = 0.0f;

    for (int kb = 0; kb < seq; kb += 32) {
        int kv_row = kb + (int)lane;
        if (kv_row < seq) {
            size_t base_k_cache = (((size_t)layer * (size_t)cache_cap) + (size_t)kv_row) * 128u + (size_t)kvh * 64u;
            size_t base_v_cache = (((size_t)layer * (size_t)cache_cap) + (size_t)kv_row) * 128u + (size_t)kvh * 64u;
            for (int d = 0; d < 64; d++) {
                shared_k[lane * 64 + d] = k_cache[base_k_cache + d];
                shared_v[lane * 64 + d] = v_cache[base_v_cache + d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query_row < seq && kb <= query_row) {
            int limit = min(32, query_row - kb + 1);
            for (int j = 0; j < limit; j++) {
                float dot = 0.0f;
                threadgroup const float *k_shared = shared_k + j * 64;
                for (int d = 0; d < 64; d++) {
                    dot += qd[d] * k_shared[d];
                }
                float p = exp(dot * scale - max_score) / denom;
                threadgroup const float *v_shared = shared_v + j * 64;
                for (int d = 0; d < 64; d++) {
                    acc[d] += p * v_shared[d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_row < seq) {
        device float *oh = out + ((size_t)query_row * (size_t)n_heads + (size_t)head) * head_dim;
        for (int d = 0; d < 64; d++) {
            oh[d] = acc[d];
        }
    }
}
