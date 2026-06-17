#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "mu.h"

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
    puts("mu_test ok");
    return 0;
}
