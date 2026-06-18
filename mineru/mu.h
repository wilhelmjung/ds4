#ifndef MU_H
#define MU_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct mu_engine mu_engine;
typedef struct mu_result mu_result;

typedef struct {
    char type[32];
    float bbox[4];
    int angle;
    bool merge_prev;
} mu_layout_block;

typedef struct {
    int grid_t;
    int grid_h;
    int grid_w;
    int rows;
    int cols;
    float *values;
} mu_image_tokens;

typedef struct {
    int id;
    float logit;
} mu_token_logit;

typedef enum {
    MU_BACKEND_CPU = 0,
    MU_BACKEND_METAL = 1,
} mu_backend;

typedef struct {
    const char *model_dir;
    mu_backend backend;
    int n_threads;
    int max_new_tokens;
    bool inspect_only;
    bool image_analysis;
    bool allow_cpu_fallback;
} mu_engine_options;

mu_engine_options mu_engine_options_default(void);

int mu_engine_open(mu_engine **out, const mu_engine_options *opt);
void mu_engine_close(mu_engine *e);
void mu_engine_summary(mu_engine *e, FILE *fp);
int mu_engine_tensor_count(const mu_engine *e);
int mu_engine_bf16_tensor_count(const mu_engine *e);
bool mu_engine_metal_available(const mu_engine *e);
int mu_engine_cpu_fallback_count(const mu_engine *e);
int mu_engine_text_layers(const mu_engine *e);
int mu_engine_hidden_size(const mu_engine *e);
int mu_engine_vision_layers(const mu_engine *e);
int mu_engine_bound_text_layers(const mu_engine *e);
int mu_engine_bound_vision_layers(const mu_engine *e);
int mu_engine_tensor_index(const mu_engine *e, const char *name);
int mu_engine_tensor_ndim(const mu_engine *e, int index);
uint64_t mu_engine_tensor_dim(const mu_engine *e, int index, int dim);
uint64_t mu_engine_tensor_size_bytes(const mu_engine *e, int index);

int mu_parse_image_file(mu_engine *e, const char *path, mu_result **out);
int mu_parse_image_rgb(mu_engine *e, const uint8_t *rgb, int width, int height,
                       int stride, mu_result **out);
void mu_result_free(mu_result *r);
int mu_result_write_json(const mu_result *r, FILE *fp);
int mu_result_write_markdown(const mu_result *r, FILE *fp);
int mu_parse_layout_markup(const char *text, mu_layout_block *blocks, int max_blocks);
char *mu_render_chat_prompt(const char *prompt, bool has_image);
int mu_tokenize_text(mu_engine *e, const char *text, int *out, int max_out);
int mu_tokenize_image_text(mu_engine *e, const char *text,
                           int grid_t, int grid_h, int grid_w,
                           int *out, int max_out);
int mu_build_position_ids(mu_engine *e, const int *input_ids, int n_ids,
                          int grid_t, int grid_h, int grid_w,
                          int *out, int max_out);
int mu_preprocess_layout_image_file(mu_engine *e, const char *path, mu_image_tokens *out);
void mu_image_tokens_free(mu_image_tokens *tokens);
int mu_vision_patch_embed(mu_engine *e, const mu_image_tokens *tokens,
                          float *out, int out_rows, int out_cols);
int mu_vision_rotary_pos_emb(mu_engine *e, int grid_t, int grid_h, int grid_w,
                             float *out, int out_rows, int out_cols);
int mu_vision_block0_norm1_token0(mu_engine *e, const float *patch_embeds,
                                  int rows, int cols, float *out, int out_n);
int mu_vision_block0_qkv_token0(mu_engine *e, const float *norm1,
                                int norm1_n, float *out, int out_n);
int mu_vision_block0_attn_token0(mu_engine *e, const float *patch_embeds,
                                 int rows, int cols,
                                 const float *rotary, int rotary_rows, int rotary_cols,
                                 float *out, int out_n);
int mu_vision_block0_output_token0(mu_engine *e, const float *patch_embeds,
                                   int rows, int cols,
                                   const float *rotary, int rotary_rows, int rotary_cols,
                                   float *out, int out_n);
int mu_vision_block0_output_token(mu_engine *e, const float *patch_embeds,
                                  int rows, int cols,
                                  const float *rotary, int rotary_rows, int rotary_cols,
                                  int token_index, float *out, int out_n);
int mu_vision_block0_output_all(mu_engine *e, const float *patch_embeds,
                                int rows, int cols,
                                const float *rotary, int rotary_rows, int rotary_cols,
                                float *out, int out_rows, int out_cols);
int mu_vision_encode(mu_engine *e, const float *patch_embeds,
                     int rows, int cols,
                     const float *rotary, int rotary_rows, int rotary_cols,
                     float *out, int out_rows, int out_cols);
int mu_vision_encode_hidden(mu_engine *e, const float *patch_embeds,
                            int rows, int cols,
                            const float *rotary, int rotary_rows, int rotary_cols,
                            float *out, int out_rows, int out_cols);
int mu_text_top_logits(mu_engine *e, const int *input_ids, int n_ids,
                       int top_k, mu_token_logit *out);
int mu_text_layer0_qkv_token0(mu_engine *e, const int *input_ids, int n_ids,
                              float *out, int out_n);
int mu_text_layer0_attn_token0(mu_engine *e, const int *input_ids, int n_ids,
                               float *out, int out_n);
int mu_text_layer0_mlp_token0(mu_engine *e, const int *input_ids, int n_ids,
                              float *out, int out_n);
int mu_text_layer0_mlp_seq(mu_engine *e, const int *input_ids, int n_ids,
                           float *out, int out_rows, int out_cols);
int mu_text_layer01_mlp_seq(mu_engine *e, const int *input_ids, int n_ids,
                            float *out, int out_rows, int out_cols);
int mu_text_generate_greedy(mu_engine *e, const int *input_ids, int n_ids,
                            int max_new_tokens, int *out);
int mu_text_top_logits_with_image_embeds(mu_engine *e, const int *input_ids, int n_ids,
                                         const int *position_ids,
                                         const float *image_embeds, int n_image_embeds,
                                         int top_k, mu_token_logit *out);
int mu_text_generate_greedy_with_image_embeds(mu_engine *e,
                                              const int *input_ids, int n_ids,
                                              int grid_t, int grid_h, int grid_w,
                                              const float *image_embeds,
                                              int n_image_embeds,
                                              int max_new_tokens, int *out);
void mu_free(void *ptr);

#endif
