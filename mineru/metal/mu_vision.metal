#include <metal_stdlib>
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

static inline float mu_rope_value(device const float *src,
                                  device const float *rope,
                                  int d) {
    float angle = rope[d < 40 ? d : d - 40];
    float c = cos(angle);
    float s = sin(angle);
    float rot = d < 40 ? -src[d + 40] : src[d - 40];
    return mu_round_bf16(src[d] * c + rot * s);
}

kernel void mu_vision_attn_concat_probe(device const float *q0 [[buffer(0)]],
                                        device const float *kv [[buffer(1)]],
                                        device const float *rotary [[buffer(2)]],
                                        device float *out [[buffer(3)]],
                                        constant int &rows [[buffer(4)]],
                                        constant int &token_index [[buffer(5)]],
                                        uint gid [[thread_position_in_grid]]) {
    const int head_dim = 80;
    const int hidden = 1280;
    if (gid >= hidden) return;

    int head = (int)gid / head_dim;
    int dim = (int)gid - head * head_dim;
    device const float *q_head = q0 + head * head_dim;
    device const float *q_rope = rotary + (size_t)token_index * 40u;
    float qd[80];
    for (int d = 0; d < head_dim; d++) {
        qd[d] = mu_rope_value(q_head, q_rope, d);
    }

    float max_score = -3.402823466e38f;
    float scale = rsqrt(80.0f);
    for (int r = 0; r < rows; r++) {
        device const float *k_head = kv + (size_t)r * 2560u + head * head_dim;
        device const float *k_rope = rotary + (size_t)r * 40u;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_rope_value(k_head, k_rope, d);
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }

    float denom = 0.0f;
    for (int r = 0; r < rows; r++) {
        device const float *k_head = kv + (size_t)r * 2560u + head * head_dim;
        device const float *k_rope = rotary + (size_t)r * 40u;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_rope_value(k_head, k_rope, d);
        }
        denom += exp(dot * scale - max_score);
    }

    float acc = 0.0f;
    for (int r = 0; r < rows; r++) {
        device const float *k_head = kv + (size_t)r * 2560u + head * head_dim;
        device const float *k_rope = rotary + (size_t)r * 40u;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_rope_value(k_head, k_rope, d);
        }
        float p = mu_round_bf16(exp(dot * scale - max_score) / denom);
        device const float *v_head = kv + (size_t)r * 2560u + 1280u + head * head_dim;
        acc += p * v_head[dim];
    }
    out[gid] = mu_round_bf16(acc);
}

kernel void mu_vision_qk_scores_head(device const float *q [[buffer(0)]],
                                     device const float *kv [[buffer(1)]],
                                     device const float *rotary [[buffer(2)]],
                                     device float *scores [[buffer(3)]],
                                     constant int &rows [[buffer(4)]],
                                     constant int &head [[buffer(5)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    int key_row = (int)gid.x;
    int query_row = (int)gid.y;
    if (key_row >= rows || query_row >= rows) return;

    const int head_dim = 80;
    device const float *q_head = q + (size_t)query_row * 1280u + head * head_dim;
    device const float *k_head = kv + (size_t)key_row * 2560u + head * head_dim;
    device const float *q_rope = rotary + (size_t)query_row * 40u;
    device const float *k_rope = rotary + (size_t)key_row * 40u;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += mu_rope_value(q_head, q_rope, d) * mu_rope_value(k_head, k_rope, d);
    }
    scores[(size_t)query_row * (size_t)rows + (size_t)key_row] = dot * rsqrt(80.0f);
}

kernel void mu_vision_rope_qk_rows(device const float *q [[buffer(0)]],
                                   device const float *kv [[buffer(1)]],
                                   device const float *rotary [[buffer(2)]],
                                   device float *q_rot [[buffer(3)]],
                                   device float *k_rot [[buffer(4)]],
                                   constant int &rows [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
    const int head_dim = 80;
    const int hidden = 1280;
    int idx = (int)gid;
    int total = rows * hidden;
    if (idx >= total) return;

    int row = idx / hidden;
    int col = idx - row * hidden;
    int head = col / head_dim;
    int dim = col - head * head_dim;
    device const float *q_head = q + (size_t)row * 1280u + head * head_dim;
    device const float *k_head = kv + (size_t)row * 2560u + head * head_dim;
    device const float *rope = rotary + (size_t)row * 40u;
    q_rot[(size_t)idx] = mu_rope_value(q_head, rope, dim);
    k_rot[(size_t)idx] = mu_rope_value(k_head, rope, dim);
}

kernel void mu_vision_qk_scores_head_prerot(device const float *q_rot [[buffer(0)]],
                                            device const float *k_rot [[buffer(1)]],
                                            device float *scores [[buffer(2)]],
                                            constant int &rows [[buffer(3)]],
                                            constant int &head [[buffer(4)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    int key_row = (int)gid.x;
    int query_row = (int)gid.y;
    if (key_row >= rows || query_row >= rows) return;

    const int head_dim = 80;
    device const float *q_head = q_rot + (size_t)query_row * 1280u + head * head_dim;
    device const float *k_head = k_rot + (size_t)key_row * 1280u + head * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += q_head[d] * k_head[d];
    }
    scores[(size_t)query_row * (size_t)rows + (size_t)key_row] = dot * rsqrt(80.0f);
}

kernel void mu_vision_softmax_bf16_rows(device float *scores [[buffer(0)]],
                                        constant int &rows [[buffer(1)]],
                                        uint row_gid [[thread_position_in_grid]]) {
    int row = (int)row_gid;
    if (row >= rows) return;
    device float *score_row = scores + (size_t)row * (size_t)rows;
    float max_score = -3.402823466e38f;
    for (int c = 0; c < rows; c++) {
        if (score_row[c] > max_score) max_score = score_row[c];
    }
    float denom = 0.0f;
    for (int c = 0; c < rows; c++) {
        denom += exp(score_row[c] - max_score);
    }
    for (int c = 0; c < rows; c++) {
        score_row[c] = mu_round_bf16(exp(score_row[c] - max_score) / denom);
    }
}

kernel void mu_vision_pv_head(device const float *scores [[buffer(0)]],
                              device const float *kv [[buffer(1)]],
                              device float *out [[buffer(2)]],
                              constant int &rows [[buffer(3)]],
                              constant int &head [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]]) {
    int dim = (int)gid.x;
    int query_row = (int)gid.y;
    const int head_dim = 80;
    if (dim >= head_dim || query_row >= rows) return;

    device const float *score_row = scores + (size_t)query_row * (size_t)rows;
    float acc = 0.0f;
    for (int key_row = 0; key_row < rows; key_row++) {
        device const float *v_head =
            kv + (size_t)key_row * 2560u + 1280u + head * head_dim;
        acc += score_row[key_row] * v_head[dim];
    }
    out[(size_t)query_row * 1280u + (size_t)head * head_dim + (size_t)dim] =
        mu_round_bf16(acc);
}

kernel void mu_vision_softmax_pv_head(device const float *scores [[buffer(0)]],
                                      device const float *kv [[buffer(1)]],
                                      device float *out [[buffer(2)]],
                                      constant int &rows [[buffer(3)]],
                                      constant int &head [[buffer(4)]],
                                      uint query_gid [[thread_position_in_grid]]) {
    int query_row = (int)query_gid;
    const int head_dim = 80;
    if (query_row >= rows) return;

    device const float *score_row = scores + (size_t)query_row * (size_t)rows;
    float max_score = -3.402823466e38f;
    for (int key_row = 0; key_row < rows; key_row++) {
        if (score_row[key_row] > max_score) max_score = score_row[key_row];
    }

    float denom = 0.0f;
    for (int key_row = 0; key_row < rows; key_row++) {
        denom += exp(score_row[key_row] - max_score);
    }

    float acc[80];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    for (int key_row = 0; key_row < rows; key_row++) {
        float p = mu_round_bf16(exp(score_row[key_row] - max_score) / denom);
        device const float *v_head =
            kv + (size_t)key_row * 2560u + 1280u + head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            acc[d] += p * v_head[d];
        }
    }

    device float *out_head = out + (size_t)query_row * 1280u + head * head_dim;
    for (int d = 0; d < head_dim; d++) {
        out_head[d] = mu_round_bf16(acc[d]);
    }
}

kernel void mu_vision_attn_rows_online(device const float *q [[buffer(0)]],
                                       device const float *kv [[buffer(1)]],
                                       device const float *rotary [[buffer(2)]],
                                       device float *out [[buffer(3)]],
                                       constant int &rows [[buffer(4)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    int query_row = (int)gid.x;
    int head = (int)gid.y;
    if (query_row >= rows || head >= 16) return;

    const int head_dim = 80;
    device const float *q_head = q + (size_t)query_row * 1280u + head * head_dim;
    device const float *q_rope = rotary + (size_t)query_row * 40u;
    float qd[80];
    for (int d = 0; d < head_dim; d++) {
        qd[d] = mu_rope_value(q_head, q_rope, d);
    }

    float scale = rsqrt(80.0f);
    float max_score = -3.402823466e38f;
    for (int key_row = 0; key_row < rows; key_row++) {
        device const float *k_head = kv + (size_t)key_row * 2560u + head * head_dim;
        device const float *k_rope = rotary + (size_t)key_row * 40u;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_rope_value(k_head, k_rope, d);
        }
        float score = dot * scale;
        if (score > max_score) max_score = score;
    }

    float denom = 0.0f;
    for (int key_row = 0; key_row < rows; key_row++) {
        device const float *k_head = kv + (size_t)key_row * 2560u + head * head_dim;
        device const float *k_rope = rotary + (size_t)key_row * 40u;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_rope_value(k_head, k_rope, d);
        }
        denom += exp(dot * scale - max_score);
    }

    float acc[80];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    for (int key_row = 0; key_row < rows; key_row++) {
        device const float *k_head = kv + (size_t)key_row * 2560u + head * head_dim;
        device const float *k_rope = rotary + (size_t)key_row * 40u;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += qd[d] * mu_rope_value(k_head, k_rope, d);
        }
        float p = mu_round_bf16(exp(dot * scale - max_score) / denom);
        device const float *v_head =
            kv + (size_t)key_row * 2560u + 1280u + head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            acc[d] += p * v_head[d];
        }
    }

    device float *out_head = out + (size_t)query_row * 1280u + head * head_dim;
    for (int d = 0; d < head_dim; d++) {
        out_head[d] = mu_round_bf16(acc[d]);
    }
}

kernel void mu_vision_add_bf16(device const float *a [[buffer(0)]],
                               device const float *b [[buffer(1)]],
                               device float *out [[buffer(2)]],
                               constant int &n [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= n) return;
    out[gid] = mu_round_bf16(a[gid] + b[gid]);
}

kernel void mu_vision_quick_gelu_bf16(device const float *x [[buffer(0)]],
                                      device float *out [[buffer(1)]],
                                      constant int &n [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= n) return;
    float v = x[gid];
    out[gid] = mu_round_bf16(v / (1.0f + exp(-1.702f * v)));
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

kernel void mu_vision_gelu_bf16(device const float *x [[buffer(0)]],
                                device float *out [[buffer(1)]],
                                constant int &n [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= n) return;
    float v = x[gid];
    out[gid] = mu_round_bf16(0.5f * v * (1.0f + mu_erf_approx(v * 0.7071067811865476f)));
}

kernel void mu_vision_merge4(device const float *hidden [[buffer(0)]],
                             device float *out [[buffer(1)]],
                             constant int &groups [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    int i = (int)gid;
    int total = groups * 5120;
    if (i >= total) return;
    int group = i / 5120;
    int col = i - group * 5120;
    int row_in_group = col / 1280;
    int inner = col - row_in_group * 1280;
    out[i] = hidden[((group * 4 + row_in_group) * 1280) + inner];
}

kernel void mu_vision_attn_rows_flash(device const float *q [[buffer(0)]],
                                      device const float *kv [[buffer(1)]],
                                      device const float *rotary [[buffer(2)]],
                                      device float *out [[buffer(3)]],
                                      constant int &rows [[buffer(4)]],
                                      uint2 tg [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    int head = (int)tg.y;
    int query_row = (int)tg.x * 32 + (int)lane;
    const int head_dim = 80;
    const float scale = rsqrt(80.0f);

    float qd[80];
    if (query_row < rows) {
        device const float *q_head = q + (size_t)query_row * 1280u + head * head_dim;
        device const float *q_rope = rotary + (size_t)query_row * 40u;
        for (int d = 0; d < 80; d++) {
            qd[d] = mu_rope_value(q_head, q_rope, d);
        }
    }

    threadgroup float shared_k[32 * 80];
    threadgroup float shared_v[32 * 80];

    float max_score = -3.402823466e38f;
    float denom = 0.0f;

    for (int kb = 0; kb < rows; kb += 32) {
        int k_row = kb + (int)lane;
        if (k_row < rows) {
            device const float *k_head = kv + (size_t)k_row * 2560u + head * head_dim;
            device const float *k_rope = rotary + (size_t)k_row * 40u;
            for (int d = 0; d < 80; d++) {
                shared_k[lane * 80 + d] = mu_rope_value(k_head, k_rope, d);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query_row < rows) {
            int limit = min(32, rows - kb);
            for (int j = 0; j < limit; j++) {
                float dot = 0.0f;
                threadgroup const float *k_head_shared = shared_k + j * 80;
                for (int d = 0; d < 80; d++) {
                    dot += qd[d] * k_head_shared[d];
                }
                float score = dot * scale;
                float m_new = max(max_score, score);
                denom = denom * exp(max_score - m_new) + exp(score - m_new);
                max_score = m_new;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float acc[80];
    for (int d = 0; d < 80; d++) {
        acc[d] = 0.0f;
    }

    for (int kb = 0; kb < rows; kb += 32) {
        int kv_row = kb + (int)lane;
        if (kv_row < rows) {
            device const float *k_head = kv + (size_t)kv_row * 2560u + head * head_dim;
            device const float *k_rope = rotary + (size_t)kv_row * 40u;
            device const float *v_head = kv + (size_t)kv_row * 2560u + 1280u + head * head_dim;
            for (int d = 0; d < 80; d++) {
                shared_k[lane * 80 + d] = mu_rope_value(k_head, k_rope, d);
                shared_v[lane * 80 + d] = v_head[d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (query_row < rows) {
            int limit = min(32, rows - kb);
            for (int j = 0; j < limit; j++) {
                float dot = 0.0f;
                threadgroup const float *k_head_shared = shared_k + j * 80;
                for (int d = 0; d < 80; d++) {
                    dot += qd[d] * k_head_shared[d];
                }
                float p = mu_round_bf16(exp(dot * scale - max_score) / denom);
                threadgroup const float *v_head_shared = shared_v + j * 80;
                for (int d = 0; d < 80; d++) {
                    acc[d] += p * v_head_shared[d];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_row < rows) {
        device float *out_head = out + (size_t)query_row * 1280u + head * head_dim;
        for (int d = 0; d < 80; d++) {
            out_head[d] = mu_round_bf16(acc[d]);
        }
    }
}

