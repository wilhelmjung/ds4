#include "mu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)len, fp);
    fclose(fp);
    if (got != (size_t)len) {
        free(buf);
        return NULL;
    }
    buf[len] = 0;
    return buf;
}

static float *read_float_file(const char *path, int count) {
    if (!path || count <= 0) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    float *buf = (float *)malloc((size_t)count * sizeof(buf[0]));
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, sizeof(buf[0]), (size_t)count, fp);
    fclose(fp);
    if (got != (size_t)count) {
        free(buf);
        return NULL;
    }
    return buf;
}

static const char *skip_json_string(const char *p) {
    if (!p || *p != '"') return p;
    p++;
    while (*p) {
        if (*p == '\\') {
            p += p[1] ? 2 : 1;
            continue;
        }
        if (*p == '"') return p + 1;
        p++;
    }
    return p;
}

static char *json_get_string(const char *json, const char *key) {
    char pattern[128];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return NULL;
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += n;
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    if (*p != '"') return NULL;
    p++;

    size_t cap = strlen(p) + 1;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t len = 0;
    while (*p) {
        if (*p == '"') {
            out[len] = 0;
            return out;
        }
        if (*p == '\\') {
            p++;
            if (!*p) break;
            if (*p == 'n') out[len++] = '\n';
            else if (*p == 'r') out[len++] = '\r';
            else if (*p == 't') out[len++] = '\t';
            else if (*p == '"' || *p == '\\' || *p == '/') out[len++] = *p;
            else {
                out[len++] = *p;
            }
            p++;
            continue;
        }
        out[len++] = *p++;
    }
    free(out);
    return NULL;
}

static int json_get_int_array(const char *json, const char *key, int **out, int *out_n) {
    char pattern[128];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return -1;
    const char *p = strstr(json, pattern);
    if (!p) return -2;
    p += n;
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    if (*p != ':') return -3;
    p++;
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    if (*p != '[') return -4;
    p++;

    int cap = 64;
    int len = 0;
    int *ids = (int *)malloc((size_t)cap * sizeof(ids[0]));
    if (!ids) return -5;
    for (;;) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (*p == ']') {
            *out = ids;
            *out_n = len;
            return 0;
        }
        char *endp = NULL;
        long v = strtol(p, &endp, 10);
        if (endp == p || v < 0 || v > 2147483647L) {
            free(ids);
            return -6;
        }
        if (len == cap) {
            cap *= 2;
            int *next = (int *)realloc(ids, (size_t)cap * sizeof(ids[0]));
            if (!next) {
                free(ids);
                return -7;
            }
            ids = next;
        }
        ids[len++] = (int)v;
        p = endp;
    }
}

static int json_get_first_ints(const char *json, const char *key, int *out, int want) {
    char pattern[128];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return -1;
    const char *p = strstr(json, pattern);
    if (!p) return -2;
    p = strchr(p + n, '[');
    if (!p) return -3;
    int got = 0;
    while (*p && got < want) {
        if ((*p >= '0' && *p <= '9') || *p == '-') {
            char *endp = NULL;
            long v = strtol(p, &endp, 10);
            if (endp == p || v < -2147483647L || v > 2147483647L) return -4;
            out[got++] = (int)v;
            p = endp;
            continue;
        }
        p++;
    }
    return got == want ? 0 : -5;
}

static int json_get_float_array(const char *json, const char *key, float **out, int *out_n) {
    char pattern[128];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return -1;
    const char *p = strstr(json, pattern);
    if (!p) return -2;
    p += n;
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    if (*p != ':') return -3;
    p++;
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    if (*p != '[') return -4;
    p++;

    int cap = 64;
    int len = 0;
    float *vals = (float *)malloc((size_t)cap * sizeof(vals[0]));
    if (!vals) return -5;
    for (;;) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (*p == ']') {
            *out = vals;
            *out_n = len;
            return 0;
        }
        char *endp = NULL;
        float v = strtof(p, &endp);
        if (endp == p) {
            free(vals);
            return -6;
        }
        if (len == cap) {
            cap *= 2;
            float *next = (float *)realloc(vals, (size_t)cap * sizeof(vals[0]));
            if (!next) {
                free(vals);
                return -7;
            }
            vals = next;
        }
        vals[len++] = v;
        p = endp;
    }
}

static const char *find_layout_blocks_array(const char *json) {
    const char *p = strstr(json, "\"layout_blocks\"");
    if (!p) return NULL;
    p = strchr(p, '[');
    return p;
}

static const char *layout_blocks_array_end(const char *array_start) {
    if (!array_start || *array_start != '[') return NULL;
    const char *p = array_start;
    int depth = 0;
    while (*p) {
        if (*p == '"') {
            p = skip_json_string(p);
            continue;
        }
        if (*p == '[') depth++;
        else if (*p == ']') {
            depth--;
            if (depth == 0) return p + 1;
        }
        p++;
    }
    return NULL;
}

static int collect_layout_types(const char *json, char types[][32], int max_types) {
    const char *start = find_layout_blocks_array(json);
    const char *end = layout_blocks_array_end(start);
    if (!start || !end) return -1;
    int n = 0;
    const char *p = start;
    while (p && p < end && n < max_types) {
        p = strstr(p, "\"type\"");
        if (!p || p >= end) break;
        const char *colon = strchr(p, ':');
        if (!colon || colon >= end) break;
        p = colon + 1;
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (*p != '"') break;
        p++;
        const char *q = p;
        while (q < end && *q && *q != '"') q++;
        size_t len = (size_t)(q - p);
        if (len >= 32) len = 31;
        memcpy(types[n], p, len);
        types[n][len] = 0;
        n++;
        p = q;
    }
    return n;
}

static const char *find_top_logits_array(const char *json) {
    const char *p = strstr(json, "\"top_logits\"");
    if (!p) return NULL;
    return strchr(p, '[');
}

static int collect_top_logits(const char *json, mu_token_logit *items, int max_items) {
    const char *start = find_top_logits_array(json);
    const char *end = layout_blocks_array_end(start);
    if (!start || !end) return -1;
    int n = 0;
    const char *p = start;
    while (p && p < end && n < max_items) {
        const char *id_key = strstr(p, "\"id\"");
        if (!id_key || id_key >= end) break;
        const char *id_colon = strchr(id_key, ':');
        if (!id_colon || id_colon >= end) break;
        char *id_end = NULL;
        long id = strtol(id_colon + 1, &id_end, 10);
        if (id_end == id_colon + 1) break;

        const char *logit_key = strstr(id_end, "\"logit\"");
        if (!logit_key || logit_key >= end) break;
        const char *logit_colon = strchr(logit_key, ':');
        if (!logit_colon || logit_colon >= end) break;
        char *logit_end = NULL;
        float logit = strtof(logit_colon + 1, &logit_end);
        if (logit_end == logit_colon + 1) break;

        items[n].id = (int)id;
        items[n].logit = logit;
        n++;
        p = logit_end;
    }
    return n;
}

static int check_layout_block0_tiny(mu_engine *engine, const char *json,
                                    const float *patch_embeds, int patch_rows,
                                    const float *rope, int rope_rows) {
    int tiny_shape[2] = {0, 0};
    char *tiny_path = json_get_string(json, "vision_block0_tiny_output_file");
    float *expected = NULL;
    float *got = NULL;
    if (!tiny_path ||
        json_get_first_ints(json, "vision_block0_tiny_output_shape", tiny_shape, 2) != 0 ||
        tiny_shape[0] <= 0 || tiny_shape[0] > patch_rows ||
        tiny_shape[0] > rope_rows || tiny_shape[1] != 1280) {
        fprintf(stderr, "trace layout vision block0 tiny output metadata is missing or invalid\n");
        free(tiny_path);
        return 1;
    }
    int n = tiny_shape[0] * tiny_shape[1];
    expected = read_float_file(tiny_path, n);
    got = (float *)malloc((size_t)n * sizeof(got[0]));
    if (!expected || !got) {
        fprintf(stderr, "trace layout vision block0 tiny output failed to load: %s\n", tiny_path);
        free(tiny_path);
        free(expected);
        free(got);
        return 1;
    }
    if (mu_vision_block0_output_all(engine, patch_embeds,
                                    tiny_shape[0], 1280,
                                    rope, tiny_shape[0], 40,
                                    got, tiny_shape[0], 1280) != 0) {
        fprintf(stderr, "trace layout vision block0 tiny output failed\n");
        free(tiny_path);
        free(expected);
        free(got);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > 0.25f) {
            fprintf(stderr, "trace layout vision block0 tiny output mismatch at %d: got %.8g expected %.8g\n",
                    i, got[i], expected[i]);
            free(tiny_path);
            free(expected);
            free(got);
            return 1;
        }
    }
    printf("trace layout vision block0 tiny output ok\n");
    free(tiny_path);
    free(expected);
    free(got);
    return 0;
}

static int check_layout_tiny_vision(mu_engine *engine, const char *json,
                                    const float *patch_embeds, int patch_rows,
                                    const float *rope, int rope_rows) {
    int tiny_shape[2] = {0, 0};
    char *tiny_path = json_get_string(json, "vision_tiny_image_embeds_file");
    int hidden_shape[2] = {0, 0};
    char *hidden_path = json_get_string(json, "vision_tiny_last_block_output_file");
    float *expected_hidden = NULL;
    float *expected = NULL;
    float *got = NULL;
    if (!tiny_path ||
        json_get_first_ints(json, "vision_tiny_image_embeds_shape", tiny_shape, 2) != 0 ||
        tiny_shape[0] <= 0 || tiny_shape[1] != 896 ||
        tiny_shape[0] * 4 > patch_rows || tiny_shape[0] * 4 > rope_rows ||
        !hidden_path ||
        json_get_first_ints(json, "vision_tiny_last_block_output_shape", hidden_shape, 2) != 0 ||
        hidden_shape[0] != tiny_shape[0] * 4 || hidden_shape[1] != 1280) {
        fprintf(stderr, "trace layout tiny vision metadata is missing or invalid\n");
        free(tiny_path);
        free(hidden_path);
        return 1;
    }
    int in_rows = tiny_shape[0] * 4;
    int n = tiny_shape[0] * tiny_shape[1];
    float *hidden = (float *)malloc((size_t)in_rows * 1280u * sizeof(hidden[0]));
    expected_hidden = read_float_file(hidden_path, in_rows * 1280);
    expected = read_float_file(tiny_path, n);
    got = (float *)malloc((size_t)n * sizeof(got[0]));
    if (!hidden || !expected_hidden || !expected || !got) {
        fprintf(stderr, "trace layout tiny vision failed to load: %s\n", tiny_path);
        free(tiny_path);
        free(hidden_path);
        free(expected_hidden);
        free(hidden);
        free(expected);
        free(got);
        return 1;
    }
    if (mu_vision_encode_hidden(engine, patch_embeds,
                                in_rows, 1280,
                                rope, in_rows, 40,
                                hidden, in_rows, 1280) != 0) {
        fprintf(stderr, "trace layout tiny vision hidden failed\n");
        free(tiny_path);
        free(hidden_path);
        free(expected_hidden);
        free(hidden);
        free(expected);
        free(got);
        return 1;
    }
    float hidden_max_diff = 0.0f;
    int hidden_max_i = 0;
    int hidden_over_05 = 0;
    int hidden_over_10 = 0;
    int hidden_n = in_rows * 1280;
    for (int i = 0; i < hidden_n; i++) {
        float diff = fabsf(hidden[i] - expected_hidden[i]);
        if (diff > hidden_max_diff) {
            hidden_max_diff = diff;
            hidden_max_i = i;
        }
        if (diff > 0.5f) hidden_over_05++;
        if (diff > 1.0f) hidden_over_10++;
    }
    float hidden_allowed = fmaxf(0.5f, fabsf(expected_hidden[hidden_max_i]) * 0.02f);
    if (hidden_max_diff > hidden_allowed) {
        fprintf(stderr,
                "trace layout tiny vision hidden mismatch max at %d: got %.8g expected %.8g diff %.8g allowed %.8g over_0.5=%d over_1.0=%d\n",
                hidden_max_i, hidden[hidden_max_i], expected_hidden[hidden_max_i],
                hidden_max_diff, hidden_allowed, hidden_over_05, hidden_over_10);
        free(tiny_path);
        free(hidden_path);
        free(expected_hidden);
        free(hidden);
        free(expected);
        free(got);
        return 1;
    }
    if (mu_vision_encode(engine, patch_embeds,
                         in_rows, 1280,
                         rope, in_rows, 40,
                         got, tiny_shape[0], 896) != 0) {
        fprintf(stderr, "trace layout tiny vision failed\n");
        free(tiny_path);
        free(hidden_path);
        free(expected_hidden);
        free(hidden);
        free(expected);
        free(got);
        return 1;
    }
    float max_diff = 0.0f;
    int max_i = 0;
    int over_05 = 0;
    int over_10 = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(got[i] - expected[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_i = i;
        }
        if (diff > 0.5f) over_05++;
        if (diff > 1.0f) over_10++;
    }
    float allowed = fmaxf(0.5f, fabsf(expected[max_i]) * 0.02f);
    if (max_diff > allowed) {
        fprintf(stderr,
                "trace layout tiny vision mismatch max at %d: got %.8g expected %.8g diff %.8g allowed %.8g over_0.5=%d over_1.0=%d\n",
                max_i, got[max_i], expected[max_i], max_diff, allowed, over_05, over_10);
        free(tiny_path);
        free(hidden_path);
        free(expected_hidden);
        free(hidden);
        free(expected);
        free(got);
        return 1;
    }
    printf("trace layout tiny vision ok\n");
    free(tiny_path);
    free(hidden_path);
    free(expected_hidden);
    free(hidden);
    free(expected);
    free(got);
    return 0;
}

static int check_trace_file(mu_engine *engine, const char *path) {
    char *json = read_file(path);
    if (!json) {
        fprintf(stderr, "failed to read trace: %s\n", path);
        return 1;
    }

    char *mode = json_get_string(json, "mode");
    char *prompt = json_get_string(json, "prompt");
    char *expected_chat = json_get_string(json, "chat_prompt");
    if (!mode || !prompt || !expected_chat) {
        fprintf(stderr, "trace is missing mode/prompt/chat_prompt\n");
        free(json);
        free(mode);
        free(prompt);
        free(expected_chat);
        return 1;
    }

    int is_layout = strcmp(mode, "layout") == 0;
    const char *trace_scope = getenv("MU_CHECK_TRACE_SCOPE");
    int text_logits_scope = trace_scope && !strcmp(trace_scope, "text-logits");
    int text_layer0_qkv_scope = trace_scope && !strcmp(trace_scope, "text-layer0-qkv");
    int text_layer0_attn_scope = trace_scope && !strcmp(trace_scope, "text-layer0-attn");
    int text_layer0_mlp_scope = trace_scope && !strcmp(trace_scope, "text-layer0-mlp");
    int text_layer0_seq_scope = trace_scope && !strcmp(trace_scope, "text-layer0-seq");
    int text_layer01_seq_scope = trace_scope && !strcmp(trace_scope, "text-layer01-seq");
    int text_layers24_seq_scope = trace_scope && !strcmp(trace_scope, "text-layers24-seq");
    int vision_block0_norm_scope = trace_scope && !strcmp(trace_scope, "vision-block0-norm");
    int vision_block0_qkv_scope = trace_scope && !strcmp(trace_scope, "vision-block0-qkv");
    int vision_block0_attn_scope = trace_scope && !strcmp(trace_scope, "vision-block0-attn");
    int vision_block0_output_scope = trace_scope && !strcmp(trace_scope, "vision-block0-output");
    int vision_block0_tiny_output_scope =
        trace_scope && !strcmp(trace_scope, "vision-block0-tiny-output");
    int layout_logits_scope = trace_scope && !strcmp(trace_scope, "layout-logits");
    char *rendered = mu_render_chat_prompt(prompt, is_layout);
    if (!rendered || strcmp(rendered, expected_chat) != 0) {
        fprintf(stderr, "trace %s chat mismatch\n", mode);
        free(json);
        free(mode);
        free(prompt);
        free(expected_chat);
        mu_free(rendered);
        return 1;
    }
    printf("trace %s chat ok\n", mode);

    int *expected_ids = NULL;
    int expected_n = 0;
    if (json_get_int_array(json, "input_ids", &expected_ids, &expected_n) != 0 || expected_n <= 0) {
        fprintf(stderr, "trace %s is missing input_ids\n", mode);
        free(json);
        free(mode);
        free(prompt);
        free(expected_chat);
        mu_free(rendered);
        free(expected_ids);
        return 1;
    }
    int cap = expected_n + 128;
    int *got_ids = (int *)malloc((size_t)cap * sizeof(got_ids[0]));
    if (!got_ids) {
        free(json);
        free(mode);
        free(prompt);
        free(expected_chat);
        mu_free(rendered);
        free(expected_ids);
        return 1;
    }
    int grid[3] = {0, 0, 0};
    int got_n = 0;
    if (is_layout) {
        if (json_get_first_ints(json, "image_grid_thw", grid, 3) != 0) {
            fprintf(stderr, "trace layout is missing image_grid_thw\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(expected_ids);
            free(got_ids);
            return 1;
        }
        got_n = mu_tokenize_image_text(engine, rendered, grid[0], grid[1], grid[2], got_ids, cap);
    } else {
        got_n = mu_tokenize_text(engine, rendered, got_ids, cap);
    }
    if (got_n != expected_n) {
        fprintf(stderr, "trace %s tokenizer length mismatch: got %d expected %d\n", mode, got_n, expected_n);
        free(json);
        free(mode);
        free(prompt);
        free(expected_chat);
        mu_free(rendered);
        free(expected_ids);
        free(got_ids);
        return 1;
    }
    for (int i = 0; i < expected_n; i++) {
        if (got_ids[i] != expected_ids[i]) {
            fprintf(stderr, "trace %s tokenizer mismatch at %d: got %d expected %d\n",
                    mode, i, got_ids[i], expected_ids[i]);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(expected_ids);
            free(got_ids);
            return 1;
        }
    }
    printf("trace %s tokenizer ok\n", mode);
    free(expected_ids);

    int *expected_pos = (int *)malloc((size_t)expected_n * 3u * sizeof(expected_pos[0]));
    int *got_pos = (int *)malloc((size_t)expected_n * 3u * sizeof(got_pos[0]));
    if (!expected_pos || !got_pos ||
        json_get_first_ints(json, "position_ids", expected_pos, expected_n * 3) != 0) {
        fprintf(stderr, "trace %s is missing position_ids\n", mode);
        free(json);
        free(mode);
        free(prompt);
        free(expected_chat);
        mu_free(rendered);
        free(got_ids);
        free(expected_pos);
        free(got_pos);
        return 1;
    }
    int pos_n = mu_build_position_ids(engine, got_ids, got_n, grid[0], grid[1], grid[2],
                                      got_pos, expected_n * 3);
    if (pos_n != expected_n * 3) {
        fprintf(stderr, "trace %s positions length mismatch: got %d expected %d\n",
                mode, pos_n, expected_n * 3);
        free(json);
        free(mode);
        free(prompt);
        free(expected_chat);
        mu_free(rendered);
        free(got_ids);
        free(expected_pos);
        free(got_pos);
        return 1;
    }
    for (int i = 0; i < expected_n * 3; i++) {
        if (got_pos[i] != expected_pos[i]) {
            fprintf(stderr, "trace %s positions mismatch at flat %d: got %d expected %d\n",
                    mode, i, got_pos[i], expected_pos[i]);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            return 1;
        }
    }
    printf("trace %s positions ok\n", mode);

    if (!is_layout && (text_layer0_qkv_scope || text_layer0_attn_scope ||
                       text_layer0_mlp_scope || text_layer0_seq_scope ||
                       text_layer01_seq_scope || text_layers24_seq_scope)) {
        float qkv[1152];
        if (mu_text_layer0_qkv_token0(engine, got_ids, got_n, qkv, 1152) != 0) {
            fprintf(stderr, "trace text layer0 qkv failed\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            return 1;
        }
        printf("trace text layer0 qkv ok\n");
        if (text_layer0_attn_scope || text_layer0_mlp_scope ||
            text_layer0_seq_scope || text_layer01_seq_scope ||
            text_layers24_seq_scope) {
            float attn_out[896];
            if (mu_text_layer0_attn_token0(engine, got_ids, got_n, attn_out, 896) != 0) {
                fprintf(stderr, "trace text layer0 attn failed\n");
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(got_ids);
                free(expected_pos);
                free(got_pos);
                return 1;
            }
            printf("trace text layer0 attn ok\n");
        }
        if (text_layer0_mlp_scope || text_layer0_seq_scope ||
            text_layer01_seq_scope || text_layers24_seq_scope) {
            float mlp_out[896];
            if (mu_text_layer0_mlp_token0(engine, got_ids, got_n, mlp_out, 896) != 0) {
                fprintf(stderr, "trace text layer0 mlp failed\n");
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(got_ids);
                free(expected_pos);
                free(got_pos);
                return 1;
            }
            printf("trace text layer0 mlp ok\n");
        }
        if (text_layer0_seq_scope || text_layer01_seq_scope || text_layers24_seq_scope) {
            float *seq_out = (float *)malloc((size_t)got_n * 896u * sizeof(seq_out[0]));
            if (!seq_out ||
                mu_text_layer0_mlp_seq(engine, got_ids, got_n,
                                       seq_out, got_n, 896) != 0) {
                fprintf(stderr, "trace text layer0 seq failed\n");
                free(seq_out);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(got_ids);
                free(expected_pos);
                free(got_pos);
                return 1;
            }
            printf("trace text layer0 seq ok\n");
            free(seq_out);
        }
        if (text_layer01_seq_scope) {
            float *seq_out = (float *)malloc((size_t)got_n * 896u * sizeof(seq_out[0]));
            if (!seq_out ||
                mu_text_layer01_mlp_seq(engine, got_ids, got_n,
                                        seq_out, got_n, 896) != 0) {
                fprintf(stderr, "trace text layer01 seq failed\n");
                free(seq_out);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(got_ids);
                free(expected_pos);
                free(got_pos);
                return 1;
            }
            printf("trace text layer01 seq ok\n");
            free(seq_out);
        }
        if (text_layers24_seq_scope) {
            float *seq_out = (float *)malloc((size_t)got_n * 896u * sizeof(seq_out[0]));
            if (!seq_out ||
                mu_text_layers_mlp_seq(engine, got_ids, got_n,
                                       24, seq_out, got_n, 896) != 0) {
                fprintf(stderr, "trace text layers24 seq failed\n");
                free(seq_out);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(got_ids);
                free(expected_pos);
                free(got_pos);
                return 1;
            }
            printf("trace text layers24 seq ok\n");
            free(seq_out);
        }
        free(json);
        free(mode);
        free(prompt);
        free(expected_chat);
        mu_free(rendered);
        free(got_ids);
        free(expected_pos);
        free(got_pos);
        return 0;
    }

    if (is_layout && !vision_block0_norm_scope && !vision_block0_qkv_scope &&
        !vision_block0_attn_scope && !vision_block0_output_scope &&
        !vision_block0_tiny_output_scope) {
        mu_token_logit expected_top[16];
        mu_token_logit got_top[16];
        int expected_top_n = collect_top_logits(json, expected_top, 16);
        char *embeds_path = json_get_string(json, "image_embeds_file");
        int embeds_shape[2] = {0, 0};
        float *expected_embed_sample = NULL;
        int expected_embed_sample_n = 0;
        float *image_embeds = NULL;
        if (expected_top_n < 8 ||
            !embeds_path ||
            json_get_first_ints(json, "image_embeds_shape", embeds_shape, 2) != 0 ||
            embeds_shape[0] <= 0 || embeds_shape[1] != 896 ||
            json_get_float_array(json, "image_embeds_sample",
                                 &expected_embed_sample, &expected_embed_sample_n) != 0) {
            fprintf(stderr, "trace layout is missing top_logits or image_embeds metadata\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            free(embeds_path);
            free(expected_embed_sample);
            return 1;
        }
        int embed_values = embeds_shape[0] * embeds_shape[1];
        image_embeds = read_float_file(embeds_path, embed_values);
        if (!image_embeds) {
            fprintf(stderr, "trace layout failed to read image embeds: %s\n", embeds_path);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            free(embeds_path);
            free(expected_embed_sample);
            return 1;
        }
        if (expected_embed_sample_n <= 0 || expected_embed_sample_n > embed_values) {
            fprintf(stderr, "trace layout image embed sample length invalid: %d\n",
                    expected_embed_sample_n);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            free(embeds_path);
            free(expected_embed_sample);
            free(image_embeds);
            return 1;
        }
        for (int i = 0; i < expected_embed_sample_n; i++) {
            float diff = fabsf(image_embeds[i] - expected_embed_sample[i]);
            if (diff > 1e-4f) {
                fprintf(stderr, "trace layout image embed sample mismatch at %d: got %.8g expected %.8g\n",
                        i, image_embeds[i], expected_embed_sample[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(got_ids);
                free(expected_pos);
                free(got_pos);
                free(embeds_path);
                free(expected_embed_sample);
                free(image_embeds);
                return 1;
            }
        }

        int got_top_n = mu_text_top_logits_with_image_embeds(engine, got_ids, got_n,
                                                            expected_pos,
                                                            image_embeds,
                                                            embeds_shape[0],
                                                            8, got_top);
        if (got_top_n < 8) {
            fprintf(stderr, "trace layout logits failed: %d\n", got_top_n);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            free(embeds_path);
            free(expected_embed_sample);
            free(image_embeds);
            return 1;
        }
        int overlap = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (got_top[i].id == expected_top[j].id) {
                    overlap++;
                    break;
                }
            }
        }
        float top_diff = fabsf(got_top[0].logit - expected_top[0].logit);
        if (got_top[0].id != expected_top[0].id || overlap < 6 || top_diff > 0.35f) {
            fprintf(stderr,
                    "trace layout logits mismatch: got top1=%d %.6g expected top1=%d %.6g overlap=%d diff=%.6g\n",
                    got_top[0].id, got_top[0].logit,
                    expected_top[0].id, expected_top[0].logit,
                    overlap, top_diff);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            free(embeds_path);
            free(expected_embed_sample);
            free(image_embeds);
            return 1;
        }
        printf("trace layout logits ok\n");
        if (layout_logits_scope) {
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            free(embeds_path);
            free(expected_embed_sample);
            free(image_embeds);
            return 0;
        }

        int *expected_layout_gen = NULL;
        int expected_layout_gen_n = 0;
        if (json_get_int_array(json, "layout_generated_ids",
                               &expected_layout_gen, &expected_layout_gen_n) != 0 ||
            expected_layout_gen_n <= 0) {
            fprintf(stderr, "trace layout is missing layout_generated_ids\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            free(embeds_path);
            free(expected_embed_sample);
            free(image_embeds);
            free(expected_layout_gen);
            return 1;
        }
        int *got_layout_gen = (int *)malloc((size_t)expected_layout_gen_n * sizeof(got_layout_gen[0]));
        if (!got_layout_gen) {
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            free(embeds_path);
            free(expected_embed_sample);
            free(image_embeds);
            free(expected_layout_gen);
            return 1;
        }
        int got_layout_gen_n = mu_text_generate_greedy_with_image_embeds(
            engine, got_ids, got_n, grid[0], grid[1], grid[2],
            image_embeds, embeds_shape[0], expected_layout_gen_n, got_layout_gen);
        if (got_layout_gen_n != expected_layout_gen_n) {
            fprintf(stderr, "trace layout generation length mismatch: got %d expected %d\n",
                    got_layout_gen_n, expected_layout_gen_n);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_pos);
            free(got_pos);
            free(embeds_path);
            free(expected_embed_sample);
            free(image_embeds);
            free(expected_layout_gen);
            free(got_layout_gen);
            return 1;
        }
        for (int i = 0; i < expected_layout_gen_n; i++) {
            if (got_layout_gen[i] != expected_layout_gen[i]) {
                fprintf(stderr, "trace layout generation mismatch at %d: got %d expected %d\n",
                        i, got_layout_gen[i], expected_layout_gen[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(got_ids);
                free(expected_pos);
                free(got_pos);
                free(embeds_path);
                free(expected_embed_sample);
                free(image_embeds);
                free(expected_layout_gen);
                free(got_layout_gen);
                return 1;
            }
        }
        printf("trace layout generation ok\n");
        free(expected_layout_gen);
        free(got_layout_gen);

        free(embeds_path);
        free(expected_embed_sample);
        free(image_embeds);
    }
    free(expected_pos);
    free(got_pos);

    if (!is_layout) {
        mu_token_logit expected_top[16];
        mu_token_logit got_top[16];
        int expected_top_n = collect_top_logits(json, expected_top, 16);
        if (expected_top_n < 8) {
            fprintf(stderr, "trace text is missing top_logits\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            return 1;
        }
        int got_top_n = mu_text_top_logits(engine, got_ids, got_n, 8, got_top);
        if (got_top_n < 8) {
            fprintf(stderr, "trace text logits failed: %d\n", got_top_n);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            return 1;
        }
        int overlap = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (got_top[i].id == expected_top[j].id) {
                    overlap++;
                    break;
                }
            }
        }
        float top_diff = fabsf(got_top[0].logit - expected_top[0].logit);
        if (got_top[0].id != expected_top[0].id || overlap < 6 || top_diff > 0.25f) {
            fprintf(stderr,
                    "trace text logits mismatch: got top1=%d %.6g expected top1=%d %.6g overlap=%d diff=%.6g\n",
                    got_top[0].id, got_top[0].logit,
                    expected_top[0].id, expected_top[0].logit,
                    overlap, top_diff);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            return 1;
        }
        printf("trace text logits ok\n");
        if (text_logits_scope) {
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            return 0;
        }

        int *expected_gen = NULL;
        int expected_gen_n = 0;
        if (json_get_int_array(json, "generated_ids", &expected_gen, &expected_gen_n) != 0 ||
            expected_gen_n <= 0) {
            fprintf(stderr, "trace text is missing generated_ids\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_gen);
            return 1;
        }
        int *got_gen = (int *)malloc((size_t)expected_gen_n * sizeof(got_gen[0]));
        if (!got_gen) {
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_gen);
            return 1;
        }
        int got_gen_n = mu_text_generate_greedy(engine, got_ids, got_n, expected_gen_n, got_gen);
        if (got_gen_n != expected_gen_n) {
            fprintf(stderr, "trace text generation length mismatch: got %d expected %d\n",
                    got_gen_n, expected_gen_n);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(got_ids);
            free(expected_gen);
            free(got_gen);
            return 1;
        }
        for (int i = 0; i < expected_gen_n; i++) {
            if (got_gen[i] != expected_gen[i]) {
                fprintf(stderr, "trace text generation mismatch at %d: got %d expected %d\n",
                        i, got_gen[i], expected_gen[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(got_ids);
                free(expected_gen);
                free(got_gen);
                return 1;
            }
        }
        printf("trace text generation ok\n");
        free(expected_gen);
        free(got_gen);
    }
    free(got_ids);

    if (is_layout) {
        int expected_shape[2] = {0, 0};
        float *expected_sample = NULL;
        int expected_sample_n = 0;
        char *image_path = json_get_string(json, "image");
        mu_image_tokens image_tokens;
        memset(&image_tokens, 0, sizeof(image_tokens));
        if (!image_path ||
            json_get_first_ints(json, "pixel_values_shape", expected_shape, 2) != 0 ||
            json_get_float_array(json, "pixel_values_sample", &expected_sample, &expected_sample_n) != 0 ||
            mu_preprocess_layout_image_file(engine, image_path, &image_tokens) != 0) {
            fprintf(stderr, "trace layout processor inputs are missing or failed\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        if (image_tokens.grid_t != grid[0] || image_tokens.grid_h != grid[1] || image_tokens.grid_w != grid[2]) {
            fprintf(stderr, "trace layout grid mismatch: got [%d,%d,%d] expected [%d,%d,%d]\n",
                    image_tokens.grid_t, image_tokens.grid_h, image_tokens.grid_w,
                    grid[0], grid[1], grid[2]);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        if (image_tokens.rows != expected_shape[0] || image_tokens.cols != expected_shape[1]) {
            fprintf(stderr, "trace layout pixel shape mismatch: got [%d,%d] expected [%d,%d]\n",
                    image_tokens.rows, image_tokens.cols, expected_shape[0], expected_shape[1]);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        int total_values = image_tokens.rows * image_tokens.cols;
        if (expected_sample_n <= 0 || expected_sample_n > total_values) {
            fprintf(stderr, "trace layout pixel sample length invalid: %d\n", expected_sample_n);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        for (int i = 0; i < expected_sample_n; i++) {
            float diff = fabsf(image_tokens.values[i] - expected_sample[i]);
            if (diff > 1e-4f) {
                fprintf(stderr, "trace layout pixel sample mismatch at %d: got %.8g expected %.8g\n",
                        i, image_tokens.values[i], expected_sample[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(image_path);
                free(expected_sample);
                mu_image_tokens_free(&image_tokens);
                return 1;
            }
        }
        printf("trace layout processor ok\n");

        int expected_patch_shape[2] = {0, 0};
        float *expected_patch_sample = NULL;
        int expected_patch_sample_n = 0;
        if (json_get_first_ints(json, "vision_patch_embeds_shape", expected_patch_shape, 2) != 0 ||
            json_get_float_array(json, "vision_patch_embeds_sample",
                                 &expected_patch_sample, &expected_patch_sample_n) != 0 ||
            expected_patch_shape[0] != image_tokens.rows ||
            expected_patch_shape[1] != 1280) {
            fprintf(stderr, "trace layout vision patch metadata is missing or invalid\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_patch_sample);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        int patch_total = expected_patch_shape[0] * expected_patch_shape[1];
        if (expected_patch_sample_n <= 0 || expected_patch_sample_n > patch_total) {
            fprintf(stderr, "trace layout vision patch sample length invalid: %d\n",
                    expected_patch_sample_n);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_patch_sample);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        float *patch_embeds = (float *)malloc((size_t)patch_total * sizeof(patch_embeds[0]));
        if (!patch_embeds ||
            mu_vision_patch_embed(engine, &image_tokens, patch_embeds,
                                  expected_patch_shape[0], expected_patch_shape[1]) != 0) {
            fprintf(stderr, "trace layout vision patch embedding failed\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_patch_sample);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        for (int i = 0; i < expected_patch_sample_n; i++) {
            float diff = fabsf(patch_embeds[i] - expected_patch_sample[i]);
            if (diff > 1e-3f) {
                fprintf(stderr, "trace layout vision patch sample mismatch at %d: got %.8g expected %.8g\n",
                        i, patch_embeds[i], expected_patch_sample[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(image_path);
                free(expected_sample);
                free(expected_patch_sample);
                free(patch_embeds);
                mu_image_tokens_free(&image_tokens);
                return 1;
            }
        }
        printf("trace layout vision patch ok\n");
        free(expected_patch_sample);

        int expected_rope_shape[2] = {0, 0};
        float *expected_rope_sample = NULL;
        int expected_rope_sample_n = 0;
        if (json_get_first_ints(json, "vision_rotary_pos_emb_shape", expected_rope_shape, 2) != 0 ||
            json_get_float_array(json, "vision_rotary_pos_emb_sample",
                                 &expected_rope_sample, &expected_rope_sample_n) != 0 ||
            expected_rope_shape[0] != image_tokens.rows ||
            expected_rope_shape[1] != 40) {
            fprintf(stderr, "trace layout vision rope metadata is missing or invalid\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_rope_sample);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        int rope_total = expected_rope_shape[0] * expected_rope_shape[1];
        if (expected_rope_sample_n <= 0 || expected_rope_sample_n > rope_total) {
            fprintf(stderr, "trace layout vision rope sample length invalid: %d\n",
                    expected_rope_sample_n);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_rope_sample);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        float *rope = (float *)malloc((size_t)rope_total * sizeof(rope[0]));
        if (!rope ||
            mu_vision_rotary_pos_emb(engine, grid[0], grid[1], grid[2], rope,
                                     expected_rope_shape[0], expected_rope_shape[1]) != 0) {
            fprintf(stderr, "trace layout vision rope failed\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_rope_sample);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        for (int i = 0; i < expected_rope_sample_n; i++) {
            float diff = fabsf(rope[i] - expected_rope_sample[i]);
            if (diff > 1e-5f) {
                fprintf(stderr, "trace layout vision rope sample mismatch at %d: got %.8g expected %.8g\n",
                        i, rope[i], expected_rope_sample[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(image_path);
                free(expected_sample);
                free(expected_rope_sample);
                free(rope);
                free(patch_embeds);
                mu_image_tokens_free(&image_tokens);
                return 1;
            }
        }
        printf("trace layout vision rope ok\n");
        free(expected_rope_sample);

        float *expected_norm_sample = NULL;
        int expected_norm_sample_n = 0;
        if (json_get_float_array(json, "vision_block0_norm1_sample",
                                 &expected_norm_sample, &expected_norm_sample_n) != 0 ||
            expected_norm_sample_n <= 0 || expected_norm_sample_n > 1280) {
            fprintf(stderr, "trace layout vision block0 norm metadata is missing or invalid\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        float norm1[1280];
        if (mu_vision_block0_norm1_token0(engine, patch_embeds,
                                          expected_patch_shape[0], expected_patch_shape[1],
                                          norm1, 1280) != 0) {
            fprintf(stderr, "trace layout vision block0 norm failed\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        for (int i = 0; i < expected_norm_sample_n; i++) {
            float diff = fabsf(norm1[i] - expected_norm_sample[i]);
            if (diff > 1e-5f) {
                fprintf(stderr, "trace layout vision block0 norm mismatch at %d: got %.8g expected %.8g\n",
                        i, norm1[i], expected_norm_sample[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(image_path);
                free(expected_sample);
                free(expected_norm_sample);
                free(rope);
                free(patch_embeds);
                mu_image_tokens_free(&image_tokens);
                return 1;
            }
        }
        printf("trace layout vision block0 norm ok\n");
        if (vision_block0_norm_scope) {
            return 0;
        }

        float *expected_qkv_sample = NULL;
        int expected_qkv_sample_n = 0;
        if (json_get_float_array(json, "vision_block0_qkv_token0_sample",
                                 &expected_qkv_sample, &expected_qkv_sample_n) != 0 ||
            expected_qkv_sample_n <= 0 || expected_qkv_sample_n > 3840) {
            fprintf(stderr, "trace layout vision block0 qkv metadata is missing or invalid\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        float *qkv = (float *)malloc(3840u * sizeof(qkv[0]));
        if (!qkv || mu_vision_block0_qkv_token0(engine, norm1, 1280, qkv, 3840) != 0) {
            fprintf(stderr, "trace layout vision block0 qkv failed\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(qkv);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        for (int i = 0; i < expected_qkv_sample_n; i++) {
            float diff = fabsf(qkv[i] - expected_qkv_sample[i]);
            if (diff > 1e-4f) {
                fprintf(stderr, "trace layout vision block0 qkv mismatch at %d: got %.8g expected %.8g\n",
                        i, qkv[i], expected_qkv_sample[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(image_path);
                free(expected_sample);
                free(expected_norm_sample);
                free(expected_qkv_sample);
                free(qkv);
                free(rope);
                free(patch_embeds);
                mu_image_tokens_free(&image_tokens);
                return 1;
            }
        }
        printf("trace layout vision block0 qkv ok\n");
        if (vision_block0_qkv_scope) {
            return 0;
        }

        float *expected_attn_sample = NULL;
        int expected_attn_sample_n = 0;
        if (json_get_float_array(json, "vision_block0_attn_output_sample",
                                 &expected_attn_sample, &expected_attn_sample_n) != 0 ||
            expected_attn_sample_n <= 0 || expected_attn_sample_n > 1280) {
            fprintf(stderr, "trace layout vision block0 attn metadata is missing or invalid\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(expected_attn_sample);
            free(qkv);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        float attn_out[1280];
        if (mu_vision_block0_attn_token0(engine, patch_embeds,
                                         expected_patch_shape[0], expected_patch_shape[1],
                                         rope, expected_rope_shape[0], expected_rope_shape[1],
                                         attn_out, 1280) != 0) {
            fprintf(stderr, "trace layout vision block0 attn failed\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(expected_attn_sample);
            free(qkv);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        for (int i = 0; i < expected_attn_sample_n; i++) {
            float diff = fabsf(attn_out[i] - expected_attn_sample[i]);
            if (diff > 0.125f) {
                fprintf(stderr, "trace layout vision block0 attn mismatch at %d: got %.8g expected %.8g\n",
                        i, attn_out[i], expected_attn_sample[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(image_path);
                free(expected_sample);
                free(expected_norm_sample);
                free(expected_qkv_sample);
                free(expected_attn_sample);
                free(qkv);
                free(rope);
                free(patch_embeds);
                mu_image_tokens_free(&image_tokens);
                return 1;
            }
        }
        printf("trace layout vision block0 attn ok\n");
        if (vision_block0_attn_scope) {
            return 0;
        }

        float *expected_block0_sample = NULL;
        int expected_block0_sample_n = 0;
        if (json_get_float_array(json, "vision_block0_output_sample",
                                 &expected_block0_sample, &expected_block0_sample_n) != 0 ||
            expected_block0_sample_n <= 0 || expected_block0_sample_n > 1280) {
            fprintf(stderr, "trace layout vision block0 output metadata is missing or invalid\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(expected_attn_sample);
            free(expected_block0_sample);
            free(qkv);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        float *expected_block0_token1_sample = NULL;
        int expected_block0_token1_sample_n = 0;
        if (json_get_float_array(json, "vision_block0_output_token1_sample",
                                 &expected_block0_token1_sample,
                                 &expected_block0_token1_sample_n) != 0 ||
            expected_block0_token1_sample_n <= 0 || expected_block0_token1_sample_n > 1280) {
            fprintf(stderr, "trace layout vision block0 token1 output metadata is missing or invalid\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(expected_attn_sample);
            free(expected_block0_sample);
            free(expected_block0_token1_sample);
            free(qkv);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        float block0_out[1280];
        if (mu_vision_block0_output_token0(engine, patch_embeds,
                                           expected_patch_shape[0], expected_patch_shape[1],
                                           rope, expected_rope_shape[0], expected_rope_shape[1],
                                           block0_out, 1280) != 0) {
            fprintf(stderr, "trace layout vision block0 output failed\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(expected_attn_sample);
            free(expected_block0_sample);
            free(expected_block0_token1_sample);
            free(qkv);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        for (int i = 0; i < expected_block0_sample_n; i++) {
            float diff = fabsf(block0_out[i] - expected_block0_sample[i]);
            if (diff > 0.25f) {
                fprintf(stderr, "trace layout vision block0 output mismatch at %d: got %.8g expected %.8g\n",
                        i, block0_out[i], expected_block0_sample[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(image_path);
                free(expected_sample);
                free(expected_norm_sample);
                free(expected_qkv_sample);
                free(expected_attn_sample);
                free(expected_block0_sample);
                free(expected_block0_token1_sample);
                free(qkv);
                free(rope);
                free(patch_embeds);
                mu_image_tokens_free(&image_tokens);
                return 1;
            }
        }
        float block0_out1[1280];
        if (mu_vision_block0_output_token(engine, patch_embeds,
                                          expected_patch_shape[0], expected_patch_shape[1],
                                          rope, expected_rope_shape[0], expected_rope_shape[1],
                                          1, block0_out1, 1280) != 0) {
            fprintf(stderr, "trace layout vision block0 token1 output failed\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(expected_attn_sample);
            free(expected_block0_sample);
            free(expected_block0_token1_sample);
            free(qkv);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        for (int i = 0; i < expected_block0_token1_sample_n; i++) {
            float diff = fabsf(block0_out1[i] - expected_block0_token1_sample[i]);
            if (diff > 0.25f) {
                fprintf(stderr, "trace layout vision block0 token1 output mismatch at %d: got %.8g expected %.8g\n",
                        i, block0_out1[i], expected_block0_token1_sample[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(image_path);
                free(expected_sample);
                free(expected_norm_sample);
                free(expected_qkv_sample);
                free(expected_attn_sample);
                free(expected_block0_sample);
                free(expected_block0_token1_sample);
                free(qkv);
                free(rope);
                free(patch_embeds);
                mu_image_tokens_free(&image_tokens);
                return 1;
            }
        }
        if (vision_block0_output_scope) {
            printf("trace layout vision block0 output ok\n");
            return 0;
        }
        if (check_layout_block0_tiny(engine, json,
                                     patch_embeds, expected_patch_shape[0],
                                     rope, expected_rope_shape[0]) != 0) {
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(expected_attn_sample);
            free(expected_block0_sample);
            free(expected_block0_token1_sample);
            free(qkv);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        if (vision_block0_tiny_output_scope) {
            return 0;
        }
        if (check_layout_tiny_vision(engine, json,
                                     patch_embeds, expected_patch_shape[0],
                                     rope, expected_rope_shape[0]) != 0) {
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(image_path);
            free(expected_sample);
            free(expected_norm_sample);
            free(expected_qkv_sample);
            free(expected_attn_sample);
            free(expected_block0_sample);
            free(expected_block0_token1_sample);
            free(qkv);
            free(rope);
            free(patch_embeds);
            mu_image_tokens_free(&image_tokens);
            return 1;
        }
        printf("trace layout vision block0 output ok\n");
        free(expected_norm_sample);
        free(expected_qkv_sample);
        free(expected_attn_sample);
        free(expected_block0_sample);
        free(expected_block0_token1_sample);
        free(qkv);
        free(rope);
        free(patch_embeds);
        free(image_path);
        free(expected_sample);
        mu_image_tokens_free(&image_tokens);

        char *raw = json_get_string(json, "layout_raw_output");
        char expected_types[128][32];
        int expected_n = collect_layout_types(json, expected_types, 128);
        if (!raw || expected_n < 0) {
            fprintf(stderr, "layout trace is missing raw output or block types\n");
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(raw);
            return 1;
        }
        mu_layout_block blocks[128];
        int got_n = mu_parse_layout_markup(raw, blocks, 128);
        if (got_n != expected_n) {
            fprintf(stderr, "layout parser count mismatch: got %d expected %d\n", got_n, expected_n);
            free(json);
            free(mode);
            free(prompt);
            free(expected_chat);
            mu_free(rendered);
            free(raw);
            return 1;
        }
        for (int i = 0; i < got_n; i++) {
            if (strcmp(blocks[i].type, expected_types[i]) != 0) {
                fprintf(stderr, "layout parser type mismatch at %d: got %s expected %s\n",
                        i, blocks[i].type, expected_types[i]);
                free(json);
                free(mode);
                free(prompt);
                free(expected_chat);
                mu_free(rendered);
                free(raw);
                return 1;
            }
        }
        printf("trace layout parser ok\n");
        free(raw);
    }

    free(json);
    free(mode);
    free(prompt);
    free(expected_chat);
    mu_free(rendered);
    return 0;
}

int main(int argc, char **argv) {
    mu_engine_options opt = mu_engine_options_default();
    const char *trace_path = NULL;
    const char *image_path = NULL;
    int write_json = 0;
    int write_markdown = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model-dir") && i + 1 < argc) {
            opt.model_dir = argv[++i];
        } else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            const char *backend = argv[++i];
            if (!strcmp(backend, "cpu")) {
                opt.backend = MU_BACKEND_CPU;
            } else if (!strcmp(backend, "metal")) {
                opt.backend = MU_BACKEND_METAL;
            } else {
                fprintf(stderr, "--backend must be cpu or metal\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--no-cpu-fallback")) {
            opt.allow_cpu_fallback = false;
        } else if (!strcmp(argv[i], "--inspect")) {
            opt.inspect_only = true;
        } else if (!strcmp(argv[i], "--check-trace") && i + 1 < argc) {
            trace_path = argv[++i];
        } else if (!strcmp(argv[i], "--image") && i + 1 < argc) {
            image_path = argv[++i];
        } else if (!strcmp(argv[i], "--json")) {
            write_json = 1;
        } else if (!strcmp(argv[i], "--markdown")) {
            write_markdown = 1;
        } else {
            fprintf(stderr, "usage: %s [--model-dir PATH] [--backend cpu|metal] [--no-cpu-fallback] [--inspect] [--check-trace PATH] [--image PATH (--json|--markdown)]\n", argv[0]);
            return 2;
        }
    }
    if (trace_path && image_path) {
        fprintf(stderr, "--check-trace and --image are mutually exclusive\n");
        return 2;
    }
    if (write_json && write_markdown) {
        fprintf(stderr, "--json and --markdown are mutually exclusive\n");
        return 2;
    }
    if ((write_json || write_markdown) && !image_path) {
        fprintf(stderr, "--json/--markdown require --image PATH\n");
        return 2;
    }
    if (image_path && !write_json && !write_markdown) {
        fprintf(stderr, "--image requires --json or --markdown\n");
        return 2;
    }

    mu_engine *engine = NULL;
    int rc = mu_engine_open(&engine, &opt);
    if (rc) {
        fprintf(stderr, "mu_engine_open failed: %d\n", rc);
        return 1;
    }
    if (trace_path) {
        rc = check_trace_file(engine, trace_path);
        mu_engine_close(engine);
        return rc;
    }
    if (image_path) {
        mu_result *result = NULL;
        rc = mu_parse_image_file(engine, image_path, &result);
        if (rc) {
            fprintf(stderr, "mu_parse_image_file failed: %d\n", rc);
            mu_result_free(result);
            mu_engine_close(engine);
            return 1;
        }
        rc = write_json ? mu_result_write_json(result, stdout)
                        : mu_result_write_markdown(result, stdout);
        if (rc == 0) fputc('\n', stdout);
        mu_result_free(result);
        mu_engine_close(engine);
        return rc == 0 ? 0 : 1;
    }
    mu_engine_summary(engine, stdout);
    mu_engine_close(engine);
    return 0;
}
