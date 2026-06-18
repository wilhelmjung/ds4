#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "mu.h"
#if defined(__APPLE__)
#include "mu_gpu.h"
#endif

extern float mu_bf16_to_f32(uint16_t v);
extern uint16_t mu_f32_to_bf16(float f);
extern float mu_silu_f32(float x);
extern void mu_dense_f32(const float *x, const uint16_t *w_bf16,
                         const float *bias, int rows, int cols, float *y);
extern void mu_rmsnorm_f32(float *x, const float *weight, int n, float eps);
extern char *mu_otsl_to_html(const char *otsl);

static int test_default_options(void) {
    mu_engine_options opt = mu_engine_options_default();
    if (opt.backend != MU_BACKEND_CPU) return 1;
    if (opt.max_new_tokens != 512) return 2;
    if (opt.n_threads < 1) return 3;
    return 0;
}

static int test_open_local_model_inspect(void) {
    mu_engine_options opt = mu_engine_options_default();
    opt.inspect_only = true;
    mu_engine *engine = NULL;
    int rc = mu_engine_open(&engine, &opt);
    if (rc) return 10 + (-rc);
    if (mu_engine_tensor_count(engine) != 681) return 20;
    if (mu_engine_bf16_tensor_count(engine) != 681) return 21;
    mu_engine_close(engine);
    return 0;
}

static int test_model_shape_constants(void) {
    mu_engine_options opt = mu_engine_options_default();
    opt.inspect_only = true;
    mu_engine *engine = NULL;
    int rc = mu_engine_open(&engine, &opt);
    if (rc) return 30 + (-rc);
    if (mu_engine_text_layers(engine) != 24) return 40;
    if (mu_engine_hidden_size(engine) != 896) return 41;
    if (mu_engine_vision_layers(engine) != 32) return 42;
    if (mu_engine_bound_text_layers(engine) != 24) return 43;
    if (mu_engine_bound_vision_layers(engine) != 32) return 44;
    mu_engine_close(engine);
    return 0;
}

static int test_tensor_table_shapes(void) {
    mu_engine_options opt = mu_engine_options_default();
    opt.inspect_only = true;
    mu_engine *engine = NULL;
    int rc = mu_engine_open(&engine, &opt);
    if (rc) return 50 + (-rc);

    int embed = mu_engine_tensor_index(engine, "model.embed_tokens.weight");
    if (embed < 0) return 60;
    if (mu_engine_tensor_ndim(engine, embed) != 2) return 61;
    if (mu_engine_tensor_dim(engine, embed, 0) != 151936) return 62;
    if (mu_engine_tensor_dim(engine, embed, 1) != 896) return 63;
    if (mu_engine_tensor_size_bytes(engine, embed) != 272269312ULL) return 64;

    int patch = mu_engine_tensor_index(engine, "visual.patch_embed.proj.weight");
    if (patch < 0) return 65;
    if (mu_engine_tensor_ndim(engine, patch) != 5) return 66;
    if (mu_engine_tensor_dim(engine, patch, 0) != 1280) return 67;
    if (mu_engine_tensor_dim(engine, patch, 1) != 3) return 68;
    if (mu_engine_tensor_dim(engine, patch, 2) != 2) return 69;
    if (mu_engine_tensor_dim(engine, patch, 3) != 14) return 70;
    if (mu_engine_tensor_dim(engine, patch, 4) != 14) return 71;

    mu_engine_close(engine);
    return 0;
}

static int test_layout_parser_basic_blocks(void) {
    const char *raw =
        "<|box_start|>10 20 300 400<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>hello\n"
        "<|box_start|>100 120 500 700<|box_end|><|ref_start|>unknown<|ref_end|><|rotate_right|>\n"
        "<|box_start|>1 1 2 2<|box_end|><|ref_start|>inline_formula<|ref_end|>";
    mu_layout_block blocks[4];
    int n = mu_parse_layout_markup(raw, blocks, 4);
    if (n != 2) return 80;
    if (strcmp(blocks[0].type, "text") != 0) return 81;
    if (blocks[0].bbox[0] < 0.009f || blocks[0].bbox[0] > 0.011f) return 82;
    if (blocks[0].bbox[1] < 0.019f || blocks[0].bbox[1] > 0.021f) return 83;
    if (blocks[0].bbox[2] < 0.299f || blocks[0].bbox[2] > 0.301f) return 84;
    if (blocks[0].bbox[3] < 0.399f || blocks[0].bbox[3] > 0.401f) return 85;
    if (blocks[0].angle != 0) return 86;
    if (strcmp(blocks[1].type, "image") != 0) return 87;
    if (blocks[1].angle != 90) return 88;
    return 0;
}

static int close_enough(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static int test_cpu_math_primitives(void) {
    if (!close_enough(mu_bf16_to_f32(0x3f80), 1.0f, 0.0f)) return 90;
    if (!close_enough(mu_bf16_to_f32(0xc000), -2.0f, 0.0f)) return 91;
    if (mu_f32_to_bf16(1.0f) != 0x3f80) return 92;
    if (mu_f32_to_bf16(-2.0f) != 0xc000) return 93;

    const float x[3] = {1.0f, 2.0f, 3.0f};
    const uint16_t w[6] = {
        0x3f80, 0x4000, 0x4040,
        0x4080, 0x40a0, 0x40c0,
    };
    float y[2] = {0.0f, 0.0f};
    mu_dense_f32(x, w, NULL, 2, 3, y);
    if (!close_enough(y[0], 14.0f, 0.0f)) return 94;
    if (!close_enough(y[1], 32.0f, 0.0f)) return 95;

    float r[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    mu_rmsnorm_f32(r, weight, 4, 0.0f);
    const float inv_rms = 1.0f / sqrtf(7.5f);
    if (!close_enough(r[0], 1.0f * inv_rms, 1e-5f)) return 96;
    if (!close_enough(r[3], 4.0f * inv_rms, 1e-5f)) return 97;

    if (!close_enough(mu_silu_f32(0.0f), 0.0f, 0.0f)) return 98;
    if (!close_enough(mu_silu_f32(1.0f), 0.7310586f, 1e-6f)) return 99;
    return 0;
}

static int test_chat_prompt_renderer(void) {
    char *text = mu_render_chat_prompt("Respond with OK if the local MinerU model is loaded.", false);
    if (!text) return 110;
    const char *expected_text =
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Respond with OK if the local MinerU model is loaded.<|im_end|>\n"
        "<|im_start|>assistant\n";
    int rc = strcmp(text, expected_text) == 0 ? 0 : 111;
    mu_free(text);
    if (rc) return rc;

    char *layout = mu_render_chat_prompt("\nLayout Detection:", true);
    if (!layout) return 112;
    const char *expected_layout =
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Layout Detection:<|im_end|>\n"
        "<|im_start|>assistant\n";
    rc = strcmp(layout, expected_layout) == 0 ? 0 : 113;
    mu_free(layout);
    return rc;
}

static int test_otsl_table_to_html(void) {
    const char *otsl =
        "<fcel>A & B<fcel><tag><nl>"
        "<fcel>wide<lcel><nl>";
    char *html = mu_otsl_to_html(otsl);
    if (!html) return 120;
    const char *expected =
        "<table><tr><td>A &amp; B</td><td>&lt;tag&gt;</td></tr>"
        "<tr><td colspan=\"2\">wide</td></tr></table>";
    int rc = strcmp(html, expected) == 0 ? 0 : 121;
    mu_free(html);
    return rc;
}

static int test_mu_gpu_dense_probe(void) {
#if defined(__APPLE__)
    mu_gpu *gpu = NULL;
    if (mu_gpu_create(&gpu) != 0) return 0;
    float x[3] = {1.0f, -2.0f, 0.5f};
    unsigned short w[6] = {
        0x3f80, 0x4000, 0x4040,
        0xbf80, 0x3f80, 0x0000,
    };
    float out[2] = {0.0f, 0.0f};
    int rc = mu_gpu_dense_probe(gpu, x, w, 2, 3, out);
    mu_gpu_destroy(gpu);
    if (rc != 0) return 130;
    if (!close_enough(out[0], -1.5f, 1e-4f)) return 131;
    if (!close_enough(out[1], -3.0f, 1e-4f)) return 132;
#endif
    return 0;
}

static int test_mu_gpu_dense_bf16_bias_probe(void) {
#if defined(__APPLE__)
    mu_gpu *gpu = NULL;
    if (mu_gpu_create(&gpu) != 0) return 0;
    float x[3] = {1.0f, -2.0f, 0.5f};
    unsigned short w[6] = {
        0x3f80, 0x4000, 0x4040,
        0xbf80, 0x3f80, 0x0000,
    };
    unsigned short b[2] = {0x3f00, 0xc000};
    float out[2] = {0.0f, 0.0f};
    int rc = mu_gpu_dense_bf16_bias_probe(gpu, x, w, b, 2, 3, out);
    mu_gpu_destroy(gpu);
    if (rc != 0) return 170;
    float exp0 = mu_bf16_to_f32(mu_f32_to_bf16(-1.5f + 0.5f));
    float exp1 = mu_bf16_to_f32(mu_f32_to_bf16(-3.0f - 2.0f));
    if (!close_enough(out[0], exp0, 1e-4f)) return 171;
    if (!close_enough(out[1], exp1, 1e-4f)) return 172;
#endif
    return 0;
}

static int test_mu_gpu_dense_bf16_bias_rows(void) {
#if defined(__APPLE__)
    mu_gpu *gpu = NULL;
    if (mu_gpu_create(&gpu) != 0) return 0;
    float x[6] = {
        1.0f, -2.0f, 0.5f,
        0.25f, 3.0f, -1.0f,
    };
    unsigned short w[6] = {
        0x3f80, 0x4000, 0x4040,
        0xbf80, 0x3f80, 0x0000,
    };
    unsigned short b[2] = {0x3f00, 0xc000};
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int rc = mu_gpu_dense_bf16_bias_rows(gpu, x, w, b, 2, 3, 2, out);
    mu_gpu_destroy(gpu);
    if (rc != 0) return 190;
    float expected[4] = {
        mu_bf16_to_f32(mu_f32_to_bf16(-1.0f)),
        mu_bf16_to_f32(mu_f32_to_bf16(-5.0f)),
        mu_bf16_to_f32(mu_f32_to_bf16(3.75f)),
        mu_bf16_to_f32(mu_f32_to_bf16(0.75f)),
    };
    for (int i = 0; i < 4; i++) {
        if (!close_enough(out[i], expected[i], 1e-4f)) return 191 + i;
    }
#endif
    return 0;
}

static int test_mu_gpu_vision_attn_concat_probe(void) {
#if defined(__APPLE__)
    mu_gpu *gpu = NULL;
    if (mu_gpu_create(&gpu) != 0) return 0;
    float *q0 = (float *)calloc(1280u, sizeof(q0[0]));
    float *kv = (float *)calloc(2u * 2560u, sizeof(kv[0]));
    float rotary[80] = {0.0f};
    float *out = (float *)calloc(1280u, sizeof(out[0]));
    if (!q0 || !kv || !out) {
        free(q0);
        free(kv);
        free(out);
        mu_gpu_destroy(gpu);
        return 180;
    }
    for (int d = 0; d < 1280; d++) {
        kv[1280 + d] = 1.0f;
        kv[2560 + 1280 + d] = 3.0f;
    }
    int rc = mu_gpu_vision_attn_concat_probe(gpu, q0, kv, rotary, 2, 0, out);
    mu_gpu_destroy(gpu);
    if (rc != 0) {
        free(q0);
        free(kv);
        free(out);
        return 181;
    }
    for (int d = 0; d < 1280; d++) {
        if (!close_enough(out[d], 2.0f, 1e-4f)) {
            free(q0);
            free(kv);
            free(out);
            return 182;
        }
    }
    free(q0);
    free(kv);
    free(out);
#endif
    return 0;
}

static int test_mu_gpu_rmsnorm_probe(void) {
#if defined(__APPLE__)
    mu_gpu *gpu = NULL;
    if (mu_gpu_create(&gpu) != 0) return 0;
    float x[4] = {1.0f, 2.0f, -3.0f, 4.0f};
    float w[4] = {1.0f, 0.5f, 2.0f, -1.0f};
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int rc = mu_gpu_rmsnorm_probe(gpu, x, w, 4, 1e-6f, out);
    mu_gpu_destroy(gpu);
    if (rc != 0) return 140;
    float ss = 1.0f + 4.0f + 9.0f + 16.0f;
    float scale = 1.0f / sqrtf(ss / 4.0f + 1e-6f);
    float exp0 = x[0] * scale * w[0];
    float exp3 = x[3] * scale * w[3];
    if (!close_enough(out[0], exp0, 1e-4f)) return 141;
    if (!close_enough(out[3], exp3, 1e-4f)) return 142;
#endif
    return 0;
}

static int test_mu_gpu_layernorm_bf16_probe(void) {
#if defined(__APPLE__)
    mu_gpu *gpu = NULL;
    if (mu_gpu_create(&gpu) != 0) return 0;
    float x[4] = {1.0f, 2.0f, -3.0f, 4.0f};
    unsigned short w[4] = {0x3f80, 0x4000, 0x3f00, 0xbf80};
    unsigned short b[4] = {0x0000, 0x3f80, 0xbf80, 0x4000};
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int rc = mu_gpu_layernorm_bf16_probe(gpu, x, w, b, 4, 1e-6f, out);
    mu_gpu_destroy(gpu);
    if (rc != 0) return 150;

    float mean = (1.0f + 2.0f - 3.0f + 4.0f) / 4.0f;
    float var = 0.0f;
    for (int i = 0; i < 4; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv = 1.0f / sqrtf(var / 4.0f + 1e-6f);
    float exp0 = mu_bf16_to_f32(mu_f32_to_bf16((x[0] - mean) * inv));
    float exp3 = mu_bf16_to_f32(mu_f32_to_bf16((x[3] - mean) * inv * -1.0f + 2.0f));
    if (!close_enough(out[0], exp0, 1e-4f)) return 151;
    if (!close_enough(out[3], exp3, 1e-4f)) return 152;
#endif
    return 0;
}

static int test_mu_gpu_layernorm_bf16_rows(void) {
#if defined(__APPLE__)
    mu_gpu *gpu = NULL;
    if (mu_gpu_create(&gpu) != 0) return 0;
    float x[8] = {
        1.0f, 2.0f, -3.0f, 4.0f,
        -1.0f, 0.0f, 3.0f, 5.0f,
    };
    unsigned short w[4] = {0x3f80, 0x4000, 0x3f00, 0xbf80};
    unsigned short b[4] = {0x0000, 0x3f80, 0xbf80, 0x4000};
    float out[8] = {0};
    int rc = mu_gpu_layernorm_bf16_rows(gpu, x, w, b, 2, 4, 1e-6f, out);
    mu_gpu_destroy(gpu);
    if (rc != 0) return 160;

    for (int row = 0; row < 2; row++) {
        const float *xr = x + row * 4;
        float mean = 0.0f;
        for (int i = 0; i < 4; i++) mean += xr[i];
        mean /= 4.0f;
        float var = 0.0f;
        for (int i = 0; i < 4; i++) {
            float d = xr[i] - mean;
            var += d * d;
        }
        float inv = 1.0f / sqrtf(var / 4.0f + 1e-6f);
        for (int i = 0; i < 4; i++) {
            float y = (xr[i] - mean) * inv;
            y = y * mu_bf16_to_f32(w[i]) + mu_bf16_to_f32(b[i]);
            float expected = mu_bf16_to_f32(mu_f32_to_bf16(y));
            if (!close_enough(out[row * 4 + i], expected, 1e-4f)) return 161 + row;
        }
    }
#endif
    return 0;
}

int main(void) {
    int rc = test_default_options();
    if (rc) {
        fprintf(stderr, "test_default_options failed: %d\n", rc);
        return rc;
    }
    rc = test_open_local_model_inspect();
    if (rc) {
        fprintf(stderr, "test_open_local_model_inspect failed: %d\n", rc);
        return rc;
    }
    rc = test_model_shape_constants();
    if (rc) {
        fprintf(stderr, "test_model_shape_constants failed: %d\n", rc);
        return rc;
    }
    rc = test_tensor_table_shapes();
    if (rc) {
        fprintf(stderr, "test_tensor_table_shapes failed: %d\n", rc);
        return rc;
    }
    rc = test_layout_parser_basic_blocks();
    if (rc) {
        fprintf(stderr, "test_layout_parser_basic_blocks failed: %d\n", rc);
        return rc;
    }
    rc = test_cpu_math_primitives();
    if (rc) {
        fprintf(stderr, "test_cpu_math_primitives failed: %d\n", rc);
        return rc;
    }
    rc = test_chat_prompt_renderer();
    if (rc) {
        fprintf(stderr, "test_chat_prompt_renderer failed: %d\n", rc);
        return rc;
    }
    rc = test_otsl_table_to_html();
    if (rc) {
        fprintf(stderr, "test_otsl_table_to_html failed: %d\n", rc);
        return rc;
    }
    rc = test_mu_gpu_dense_probe();
    if (rc) {
        fprintf(stderr, "test_mu_gpu_dense_probe failed: %d\n", rc);
        return rc;
    }
    rc = test_mu_gpu_dense_bf16_bias_probe();
    if (rc) {
        fprintf(stderr, "test_mu_gpu_dense_bf16_bias_probe failed: %d\n", rc);
        return rc;
    }
    rc = test_mu_gpu_dense_bf16_bias_rows();
    if (rc) {
        fprintf(stderr, "test_mu_gpu_dense_bf16_bias_rows failed: %d\n", rc);
        return rc;
    }
    rc = test_mu_gpu_vision_attn_concat_probe();
    if (rc) {
        fprintf(stderr, "test_mu_gpu_vision_attn_concat_probe failed: %d\n", rc);
        return rc;
    }
    rc = test_mu_gpu_rmsnorm_probe();
    if (rc) {
        fprintf(stderr, "test_mu_gpu_rmsnorm_probe failed: %d\n", rc);
        return rc;
    }
    rc = test_mu_gpu_layernorm_bf16_probe();
    if (rc) {
        fprintf(stderr, "test_mu_gpu_layernorm_bf16_probe failed: %d\n", rc);
        return rc;
    }
    rc = test_mu_gpu_layernorm_bf16_rows();
    if (rc) {
        fprintf(stderr, "test_mu_gpu_layernorm_bf16_rows failed: %d\n", rc);
        return rc;
    }
    puts("mu_test ok");
    return 0;
}
