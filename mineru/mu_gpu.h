#ifndef MU_GPU_H
#define MU_GPU_H

#include <stdbool.h>

typedef struct mu_gpu mu_gpu;
typedef struct mu_gpu_kv_cache mu_gpu_kv_cache;

int mu_gpu_create(mu_gpu **out);
void mu_gpu_destroy(mu_gpu *gpu);
bool mu_gpu_available(const mu_gpu *gpu);
const char *mu_gpu_device_name(const mu_gpu *gpu);
int mu_gpu_dense_probe(mu_gpu *gpu, const float *x,
                       const unsigned short *w_bf16,
                       int rows, int cols, float *out);
int mu_gpu_dense_bf16_bias_probe(mu_gpu *gpu, const float *x,
                                 const unsigned short *w_bf16,
                                 const unsigned short *bias_bf16,
                                 int rows, int cols, float *out);
int mu_gpu_dense_f32_bias_probe(mu_gpu *gpu, const float *x,
                                const unsigned short *w_bf16,
                                const unsigned short *bias_bf16,
                                int rows, int cols, float *out);
int mu_gpu_dense_f32_rows(mu_gpu *gpu, const float *x,
                          const unsigned short *w_bf16,
                          int x_rows, int cols, int out_cols,
                          float *out);
int mu_gpu_dense_f32_bias_rows(mu_gpu *gpu, const float *x,
                               const unsigned short *w_bf16,
                               const unsigned short *bias_bf16,
                               int x_rows, int cols, int out_cols,
                               float *out);
int mu_gpu_dense_bf16_bias_rows(mu_gpu *gpu, const float *x,
                                const unsigned short *w_bf16,
                                const unsigned short *bias_bf16,
                                int x_rows, int cols, int out_cols,
                                float *out);
int mu_gpu_rmsnorm_probe(mu_gpu *gpu, const float *x, const float *weight,
                         int n, float eps, float *out);
int mu_gpu_rmsnorm_bf16_probe(mu_gpu *gpu, const float *x,
                              const unsigned short *weight_bf16,
                              int n, float eps, float *out);
int mu_gpu_rmsnorm_bf16_rows(mu_gpu *gpu, const float *x,
                             const unsigned short *weight_bf16,
                             int rows, int cols, float eps, float *out);
int mu_gpu_layernorm_bf16_probe(mu_gpu *gpu, const float *x,
                                const unsigned short *weight_bf16,
                                const unsigned short *bias_bf16,
                                int n, float eps, float *out);
int mu_gpu_layernorm_bf16_rows(mu_gpu *gpu, const float *x,
                               const unsigned short *weight_bf16,
                               const unsigned short *bias_bf16,
                               int rows, int cols, float eps, float *out);
int mu_gpu_vision_attn_concat_probe(mu_gpu *gpu, const float *q0,
                                    const float *kv, const float *rotary,
                                    int rows, int token_index, float *out);
int mu_gpu_vision_attn_rows(mu_gpu *gpu, const float *q,
                            const float *kv, const float *rotary,
                            int rows, float *out);
int mu_gpu_text_attn_token0(mu_gpu *gpu, const float *v, float *out);
int mu_gpu_text_attn_seq(mu_gpu *gpu, const float *q, const float *k,
                         const float *v, int seq, float *out);
int mu_gpu_text_attn_seq_pos(mu_gpu *gpu, const float *q, const float *k,
                             const float *v, const int *position_ids,
                             int seq, float *out);
int mu_gpu_text_attn_cached(mu_gpu *gpu, const float *q,
                            const float *k_cache, const float *v_cache,
                            int cache_len, float *out);
int mu_gpu_add_f32(mu_gpu *gpu, const float *a, const float *b,
                   int n, float *out);
int mu_gpu_silu_mul_f32(mu_gpu *gpu, const float *gate, const float *up,
                        int n, float *out);
int mu_gpu_vision_add_bf16(mu_gpu *gpu, const float *a, const float *b,
                           int n, float *out);
int mu_gpu_vision_quick_gelu_bf16(mu_gpu *gpu, const float *x,
                                  int n, float *out);
int mu_gpu_vision_gelu_bf16(mu_gpu *gpu, const float *x,
                            int n, float *out);
int mu_gpu_vision_merge4(mu_gpu *gpu, const float *hidden,
                         int rows, float *out);
int mu_gpu_vision_encode(mu_gpu *gpu, void *engine,
                         const float *patch_embeds,
                         int rows, int cols,
                         const float *rotary,
                         int rotary_rows, int rotary_cols,
                         float *out, int out_rows, int out_cols);

typedef struct mu_gpu_cmd_ctx mu_gpu_cmd_ctx;

int mu_gpu_cmd_begin(mu_gpu *gpu, mu_gpu_cmd_ctx **ctx);
int mu_gpu_cmd_begin_with_scratch_offsets(mu_gpu *gpu, unsigned long offset_a,
                                          unsigned long offset_b,
                                          mu_gpu_cmd_ctx **ctx);
void mu_gpu_cmd_set_label(mu_gpu_cmd_ctx *ctx, const char *label);
void mu_gpu_cmd_get_scratch_offsets(mu_gpu_cmd_ctx *ctx, unsigned long *offset_a,
                                    unsigned long *offset_b);
int mu_gpu_cmd_commit_and_wait(mu_gpu_cmd_ctx *ctx);
void mu_gpu_cmd_discard(mu_gpu_cmd_ctx *ctx);

typedef struct {
    void *ptr;
    unsigned long offset;
} mu_gpu_buf;

mu_gpu_buf mu_gpu_get_weight_buf(mu_gpu *gpu, const void *cpu_ptr, unsigned long length);
mu_gpu_buf mu_gpu_scratch_b_at(mu_gpu *gpu, unsigned long offset, unsigned long size);
mu_gpu_buf mu_gpu_scratch_alloc_a_ctx(mu_gpu_cmd_ctx *ctx, unsigned long size);
mu_gpu_buf mu_gpu_scratch_alloc_b_ctx(mu_gpu_cmd_ctx *ctx, unsigned long size);
void mu_gpu_buf_copy_to(mu_gpu_buf dst, const void *src, unsigned long size);
void mu_gpu_buf_copy_from(void *dst, mu_gpu_buf src, unsigned long size);

// C-compatible _ctx operator signatures
int mu_gpu_rmsnorm_bf16_probe_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf weight,
                                  mu_gpu_buf out, int n, float eps);
int mu_gpu_dense_f32_bias_probe_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                    mu_gpu_buf bias, mu_gpu_buf out, int rows, int cols);
int mu_gpu_text_decode_qkv_proj_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x,
                                    mu_gpu_buf qw, mu_gpu_buf qb,
                                    mu_gpu_buf kw, mu_gpu_buf kb,
                                    mu_gpu_buf vw, mu_gpu_buf vb,
                                    mu_gpu_buf q_out, mu_gpu_buf k_out, mu_gpu_buf v_out,
                                    int cols);
int mu_gpu_text_decode_qkv_rope_cache_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x,
                                          mu_gpu_buf qw, mu_gpu_buf qb,
                                          mu_gpu_buf kw, mu_gpu_buf kb,
                                          mu_gpu_buf vw, mu_gpu_buf vb,
                                          mu_gpu_buf q_out,
                                          mu_gpu_kv_cache *cache, int layer,
                                          int cache_pos, const int pos3[3],
                                          int cols);
int mu_gpu_dense_probe_add_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                               mu_gpu_buf residual, mu_gpu_buf out, int rows, int cols);
int mu_gpu_dense_probe_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                           mu_gpu_buf out, int rows, int cols);
int mu_gpu_add_f32_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf a, mu_gpu_buf b,
                       mu_gpu_buf out, int n);
int mu_gpu_silu_mul_f32_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf gate, mu_gpu_buf up,
                            mu_gpu_buf out, int n);
int mu_gpu_text_attn_cached_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q,
                                const float *k_cache, const float *v_cache,
                                int cache_len, mu_gpu_buf out);
int mu_gpu_layernorm_bf16_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf weight,
                                   mu_gpu_buf bias, int rows, int cols, float eps, mu_gpu_buf out);
int mu_gpu_rmsnorm_bf16_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf weight,
                                 int rows, int cols, float eps, mu_gpu_buf out);
int mu_gpu_dense_bf16_bias_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                    mu_gpu_buf bias, int x_rows, int cols, int out_cols, mu_gpu_buf out);
int mu_gpu_dense_bf16_bias_rows_quick_gelu_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                               mu_gpu_buf bias, int x_rows, int cols, int out_cols, mu_gpu_buf out);
int mu_gpu_dense_bf16_bias_rows_gelu_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                         mu_gpu_buf bias, int x_rows, int cols, int out_cols, mu_gpu_buf out);
int mu_gpu_text_decode_fused_ffn_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf hs_in,
                                     mu_gpu_buf post_norm_w, mu_gpu_buf gate_w,
                                     mu_gpu_buf up_w, mu_gpu_buf down_w,
                                     float eps, mu_gpu_buf out);
int mu_gpu_dense_f32_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                              int x_rows, int cols, int out_cols, mu_gpu_buf out);
int mu_gpu_dense_f32_bias_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x, mu_gpu_buf w,
                                   mu_gpu_buf bias, int x_rows, int cols, int out_cols, mu_gpu_buf out);
int mu_gpu_text_attn_seq_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q, mu_gpu_buf k,
                             mu_gpu_buf v, int seq, mu_gpu_buf out);
int mu_gpu_text_attn_seq_pos_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q, mu_gpu_buf k,
                                 mu_gpu_buf v, mu_gpu_buf position_ids, int seq, mu_gpu_buf out);
int mu_gpu_vision_attn_concat_probe_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q0, mu_gpu_buf kv,
                                        const float *rotary, int rows, int token_index, mu_gpu_buf out);
int mu_gpu_vision_attn_rows_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q, mu_gpu_buf kv,
                                const float *rotary, int rows, mu_gpu_buf out);
int mu_gpu_vision_add_bf16_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf a, mu_gpu_buf b,
                               int n, mu_gpu_buf out);
int mu_gpu_vision_quick_gelu_bf16_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x,
                                      int n, mu_gpu_buf out);
int mu_gpu_vision_gelu_bf16_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf x,
                                 int n, mu_gpu_buf out);
int mu_gpu_vision_merge4_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf hidden,
                             int rows, mu_gpu_buf out);

int mu_gpu_kv_cache_create(mu_gpu *gpu, int layers, int cap, mu_gpu_kv_cache **out);
void mu_gpu_kv_cache_destroy(mu_gpu_kv_cache *cache);
int mu_gpu_kv_cache_update_layer(mu_gpu_kv_cache *cache, int layer, int pos, const float *k_val, const float *v_val);
int mu_gpu_kv_cache_upload_all(mu_gpu_kv_cache *cache, const float *k_cpu, const float *v_cpu);
int mu_gpu_text_attn_cached_resident_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q,
                                         mu_gpu_kv_cache *cache, int layer,
                                         int cache_len, mu_gpu_buf out);
int mu_gpu_text_rope_cache_update_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf q,
                                      mu_gpu_buf k, mu_gpu_buf v,
                                      mu_gpu_kv_cache *cache, int layer,
                                      int cache_pos, const int pos3[3]);

int mu_gpu_text_logits_argmax_ctx(mu_gpu_cmd_ctx *ctx, mu_gpu_buf hidden_state,
                                  mu_gpu_buf final_norm_bf16,
                                  mu_gpu_buf embed_bf16,
                                  float eps, int hidden_dim, int vocab_dim,
                                  mu_gpu_buf out_id, mu_gpu_buf out_val);
int mu_gpu_text_logits_argmax(mu_gpu *gpu, const float *hidden_state_cpu,
                              const unsigned short *final_norm_bf16,
                              const unsigned short *embed_bf16,
                              float eps, int hidden_dim, int vocab_dim,
                              int *out_id, float *out_val);

const unsigned short *mu_engine_get_vision_block_tensor(void *engine, int layer, const char *suffix, int ndim, unsigned long d0, unsigned long d1);
const unsigned short *mu_engine_get_vision_merger_tensor(void *engine, const char *name, int ndim, unsigned long d0, unsigned long d1);
const unsigned short *mu_engine_get_text_layer_tensor(void *engine, int layer, const char *suffix, int ndim, unsigned long d0, unsigned long d1);
int mu_gpu_text_layers_mlp_seq(mu_gpu *gpu, void *engine,
                               const int *input_ids, int n_ids,
                               int n_layers, float *out);
int mu_gpu_text_layers_mlp_seq_from_hidden(mu_gpu *gpu, void *engine,
                                           const float *initial_hidden,
                                           int n_ids, const int *position_ids,
                                           int n_layers, float *out);
int mu_gpu_text_prefill_cache_from_embeddings(mu_gpu *gpu, void *engine,
                                              const float *hidden_states,
                                              int n_ids, const int *position_ids,
                                              int cache_cap,
                                              mu_gpu_kv_cache *gpu_cache,
                                              float *last_hidden_out);

#endif
