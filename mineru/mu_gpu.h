#ifndef MU_GPU_H
#define MU_GPU_H

#include <stdbool.h>

typedef struct mu_gpu mu_gpu;

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

#endif
