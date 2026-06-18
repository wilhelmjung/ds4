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
