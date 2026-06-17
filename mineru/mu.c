#include "mu.h"

#include <errno.h>
#include <ctype.h>
#include <float.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#endif

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

#ifdef __APPLE__
#include "mu_gpu.h"
#endif

static void mu_debug_dump_rgb(const char *prefix, const unsigned char *rgb, int width, int height);

typedef struct {
    int fd;
    const unsigned char *map;
    uint64_t size;
    uint64_t header_len;
    const char *header_json;
} mu_safetensors;

typedef struct {
    int text_layers;
    int hidden_size;
    int text_heads;
    int kv_heads;
    int intermediate_size;
    int vocab_size;
    int vision_layers;
    int vision_embed_dim;
    int vision_heads;
    int patch_size;
    int temporal_patch_size;
    int spatial_merge_size;
} mu_config;

typedef struct {
    char *name;
    int ndim;
    uint64_t shape[8];
    uint64_t data_offset;
    uint64_t data_end;
    uint64_t nbytes;
    const void *data;
} mu_tensor;

typedef struct {
    char *key;
    int value;
} mu_hash_slot;

typedef struct {
    mu_hash_slot *slots;
    size_t cap;
    size_t count;
} mu_hash;

typedef struct {
    const char *text;
    int id;
} mu_special_token;

struct mu_engine {
    mu_engine_options opt;
    mu_safetensors st;
    mu_config cfg;
    mu_tensor *tensors;
    mu_hash vocab;
    mu_hash merges;
    char *byte_encoder[256];
    int tokenizer_loaded;
    int tensor_count;
    int bf16_tensor_count;
    int bound_text_layers;
    int bound_vision_layers;
    bool metal_available;
    int cpu_fallback_count;
#ifdef __APPLE__
    mu_gpu *gpu;
#endif
};

struct mu_result {
    char *json;
    char *markdown;
};

static int mu_record_cpu_fallback(mu_engine *e, const char *stage) {
    if (!e) return -1;
    if (e->opt.backend != MU_BACKEND_METAL) return 0;
    if (!e->opt.allow_cpu_fallback) {
        fprintf(stderr, "mu metal stage unavailable without CPU fallback: %s\n",
                stage ? stage : "unknown");
        return -20;
    }
    e->cpu_fallback_count++;
    if (getenv("MU_METAL_DEBUG")) {
        fprintf(stderr, "mu metal fallback to CPU: %s\n",
                stage ? stage : "unknown");
    }
    return 0;
}

static uint64_t mu_read_u64_le(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) {
        v = (v << 8) | p[i];
    }
    return v;
}

float mu_bf16_to_f32(uint16_t v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t mu_f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}

float mu_silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

void mu_dense_f32(const float *x, const uint16_t *w_bf16,
                  const float *bias, int rows, int cols, float *y) {
    for (int r = 0; r < rows; r++) {
        float sum = bias ? bias[r] : 0.0f;
        const uint16_t *wrow = w_bf16 + (uint64_t)r * (uint64_t)cols;
        for (int c = 0; c < cols; c++) {
            sum += x[c] * mu_bf16_to_f32(wrow[c]);
        }
        y[r] = sum;
    }
}

void mu_rmsnorm_f32(float *x, const float *weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * inv * weight[i];
}

static int mu_join_path(char *dst, size_t cap, const char *dir, const char *name) {
    if (!dst || !cap || !dir || !name) return -1;
    int n = snprintf(dst, cap, "%s/%s", dir, name);
    if (n < 0 || (size_t)n >= cap) return -2;
    return 0;
}

static void mu_safetensors_close(mu_safetensors *st) {
    if (!st) return;
    if (st->map && st->map != MAP_FAILED) {
        munmap((void *)st->map, (size_t)st->size);
    }
    if (st->fd >= 0) close(st->fd);
    memset(st, 0, sizeof(*st));
    st->fd = -1;
}

static int mu_safetensors_open(mu_safetensors *st, const char *path) {
    if (!st || !path) return -1;
    memset(st, 0, sizeof(*st));
    st->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -2;

    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        close(fd);
        return -3;
    }
    if (sb.st_size < 16) {
        close(fd);
        return -4;
    }

    void *map = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return -5;
    }

    st->fd = fd;
    st->map = (const unsigned char *)map;
    st->size = (uint64_t)sb.st_size;
    st->header_len = mu_read_u64_le(st->map);
    if (st->header_len == 0 || 8 + st->header_len > st->size) {
        mu_safetensors_close(st);
        return -6;
    }
    st->header_json = (const char *)(st->map + 8);
    return 0;
}

static const char *mu_json_skip_string(const char *p, const char *end) {
    if (p >= end || *p != '"') return p;
    p++;
    while (p < end) {
        if (*p == '\\') {
            p += (p + 1 < end) ? 2 : 1;
            continue;
        }
        if (*p == '"') return p + 1;
        p++;
    }
    return end;
}

static int mu_json_key_equals(const char *start, const char *end, const char *lit) {
    size_t len = (size_t)(end - start);
    return strlen(lit) == len && memcmp(start, lit, len) == 0;
}

static char *mu_strndup_c99(const char *s, size_t len) {
    char *out = (char *)malloc(len + 1);
    if (!out) return NULL;
    memcpy(out, s, len);
    out[len] = 0;
    return out;
}

static char *mu_strdup_c99(const char *s) {
    return mu_strndup_c99(s, strlen(s));
}

static uint64_t mu_hash_string(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

static int mu_hash_init(mu_hash *h, size_t cap) {
    h->slots = (mu_hash_slot *)calloc(cap, sizeof(h->slots[0]));
    if (!h->slots) return -1;
    h->cap = cap;
    h->count = 0;
    return 0;
}

static void mu_hash_free(mu_hash *h) {
    if (!h || !h->slots) return;
    for (size_t i = 0; i < h->cap; i++) free(h->slots[i].key);
    free(h->slots);
    memset(h, 0, sizeof(*h));
}

static int mu_hash_put(mu_hash *h, const char *key, int value) {
    if (!h || !h->slots || !key) return -1;
    if ((h->count + 1) * 10 >= h->cap * 7) return -2;
    uint64_t hv = mu_hash_string(key);
    size_t pos = (size_t)(hv & (h->cap - 1));
    for (;;) {
        if (!h->slots[pos].key) {
            h->slots[pos].key = mu_strdup_c99(key);
            if (!h->slots[pos].key) return -3;
            h->slots[pos].value = value;
            h->count++;
            return 0;
        }
        if (!strcmp(h->slots[pos].key, key)) {
            h->slots[pos].value = value;
            return 0;
        }
        pos = (pos + 1) & (h->cap - 1);
    }
}

static int mu_hash_get(const mu_hash *h, const char *key, int *value) {
    if (!h || !h->slots || !key) return 0;
    uint64_t hv = mu_hash_string(key);
    size_t pos = (size_t)(hv & (h->cap - 1));
    for (;;) {
        if (!h->slots[pos].key) return 0;
        if (!strcmp(h->slots[pos].key, key)) {
            if (value) *value = h->slots[pos].value;
            return 1;
        }
        pos = (pos + 1) & (h->cap - 1);
    }
}

static const char *mu_skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
    return p;
}

static const char *mu_json_skip_value(const char *p, const char *end) {
    p = mu_skip_ws(p, end);
    if (p >= end) return end;
    if (*p == '"') return mu_json_skip_string(p, end);
    if (*p == '{' || *p == '[') {
        char open = *p;
        char close = open == '{' ? '}' : ']';
        int depth = 0;
        while (p < end) {
            if (*p == '"') {
                p = mu_json_skip_string(p, end);
                continue;
            }
            if (*p == open) depth++;
            else if (*p == close) {
                depth--;
                if (depth == 0) return p + 1;
            }
            p++;
        }
        return end;
    }
    while (p < end && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

static int mu_count_pattern(const char *hay, uint64_t len, const char *needle) {
    size_t nlen = strlen(needle);
    int count = 0;
    if (!hay || !needle || !nlen || len < nlen) return 0;
    for (uint64_t i = 0; i + nlen <= len; i++) {
        if (memcmp(hay + i, needle, nlen) == 0) count++;
    }
    return count;
}

static int mu_header_has_tensor(const mu_safetensors *st, const char *name) {
    char pattern[512];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\":", name);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return 0;
    return mu_count_pattern(st->header_json, st->header_len, pattern) > 0;
}

static const char *mu_find_bounded(const char *p, const char *end, const char *needle) {
    size_t nlen = strlen(needle);
    if (!nlen) return p;
    while (p + nlen <= end) {
        if (memcmp(p, needle, nlen) == 0) return p;
        p++;
    }
    return NULL;
}

static int mu_parse_u64_manual(const char **pp, const char *end, uint64_t *out) {
    const char *p = mu_skip_ws(*pp, end);
    uint64_t v = 0;
    int seen = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        seen = 1;
        v = v * 10 + (uint64_t)(*p - '0');
        p++;
    }
    if (!seen) return -1;
    *out = v;
    *pp = p;
    return 0;
}

static int mu_parse_u64_array_field(const char *obj, const char *obj_end,
                                    const char *field, uint64_t *vals,
                                    int max_vals, int *out_count) {
    char pattern[128];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\":[", field);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return -1;
    const char *p = mu_find_bounded(obj, obj_end, pattern);
    if (!p) return -2;
    p += n;
    int count = 0;
    while (p < obj_end) {
        p = mu_skip_ws(p, obj_end);
        if (*p == ']') {
            *out_count = count;
            return 0;
        }
        if (count >= max_vals) return -3;
        if (mu_parse_u64_manual(&p, obj_end, &vals[count]) != 0) return -4;
        count++;
        p = mu_skip_ws(p, obj_end);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == ']') {
            *out_count = count;
            return 0;
        }
        return -5;
    }
    return -6;
}

static int mu_parse_tensor_object(mu_engine *e, mu_tensor *t,
                                  const char *name_start, const char *name_end,
                                  const char *obj, const char *obj_end) {
    if (!mu_find_bounded(obj, obj_end, "\"dtype\":\"BF16\"")) return -1;
    t->name = mu_strndup_c99(name_start, (size_t)(name_end - name_start));
    if (!t->name) return -2;
    int nshape = 0;
    if (mu_parse_u64_array_field(obj, obj_end, "shape", t->shape, 8, &nshape) != 0) return -3;
    if (nshape <= 0) return -4;
    t->ndim = nshape;
    uint64_t offsets[2] = {0, 0};
    int noff = 0;
    if (mu_parse_u64_array_field(obj, obj_end, "data_offsets", offsets, 2, &noff) != 0) return -5;
    if (noff != 2 || offsets[1] < offsets[0]) return -6;
    uint64_t data_base = 8 + e->st.header_len;
    if (data_base + offsets[1] > e->st.size) return -7;
    t->data_offset = offsets[0];
    t->data_end = offsets[1];
    t->nbytes = offsets[1] - offsets[0];
    t->data = e->st.map + data_base + offsets[0];
    e->bf16_tensor_count++;
    return 0;
}

static int mu_safetensors_parse_tensors(mu_engine *e) {
    const char *p = e->st.header_json;
    const char *end = e->st.header_json + e->st.header_len;
    p = mu_skip_ws(p, end);
    if (p >= end || *p != '{') return -1;
    p++;

    e->tensors = (mu_tensor *)calloc((size_t)e->tensor_count, sizeof(e->tensors[0]));
    if (!e->tensors) return -2;

    int idx = 0;
    while (p < end) {
        p = mu_skip_ws(p, end);
        if (p < end && *p == ',') {
            p++;
            continue;
        }
        p = mu_skip_ws(p, end);
        if (p < end && *p == '}') break;
        if (p >= end || *p != '"') return -3;

        const char *key_start = p + 1;
        const char *after_key = mu_json_skip_string(p, end);
        if (after_key <= key_start) return -4;
        const char *key_end = after_key - 1;
        p = mu_skip_ws(after_key, end);
        if (p >= end || *p != ':') return -5;
        p++;
        p = mu_skip_ws(p, end);

        if (mu_json_key_equals(key_start, key_end, "__metadata__")) {
            p = mu_json_skip_value(p, end);
            continue;
        }

        if (idx >= e->tensor_count) return -6;
        if (p >= end || *p != '{') return -7;
        const char *obj = p;
        const char *obj_end = mu_json_skip_value(p, end);
        if (obj_end <= obj || obj_end > end) return -8;
        int rc = mu_parse_tensor_object(e, &e->tensors[idx], key_start, key_end, obj, obj_end);
        if (rc) return -20 + rc;
        idx++;
        p = obj_end;
    }
    if (idx != e->tensor_count) return -9;
    return 0;
}

static int mu_safetensors_count_top_level_tensors(const mu_safetensors *st) {
    if (!st || !st->header_json || st->header_len < 2) return -1;
    const char *p = st->header_json;
    const char *end = st->header_json + st->header_len;
    int depth = 0;
    int count = 0;

    while (p < end) {
        char c = *p;
        if (c == '"') {
            if (depth == 1) {
                const char *key_start = p + 1;
                const char *after = mu_json_skip_string(p, end);
                if (after <= end && after > key_start) {
                    const char *key_end = after - 1;
                    const char *q = after;
                    while (q < end && (*q == ' ' || *q == '\n' || *q == '\r' || *q == '\t')) q++;
                    if (q < end && *q == ':') {
                        if (!mu_json_key_equals(key_start, key_end, "__metadata__")) {
                            count++;
                        }
                        p = q + 1;
                        continue;
                    }
                }
                p = after;
                continue;
            }
            p = mu_json_skip_string(p, end);
            continue;
        }
        if (c == '{' || c == '[') depth++;
        else if (c == '}' || c == ']') depth--;
        if (depth < 0) return -2;
        p++;
    }
    return count;
}

static int mu_load_safetensors(mu_engine *e) {
    char path[4096];
    int rc = mu_join_path(path, sizeof(path), e->opt.model_dir, "model.safetensors");
    if (rc) return rc;
    rc = mu_safetensors_open(&e->st, path);
    if (rc) return rc;
    e->tensor_count = mu_safetensors_count_top_level_tensors(&e->st);
    if (e->tensor_count < 0) return -20;
    rc = mu_safetensors_parse_tensors(e);
    if (rc) return rc;
    if (e->tensor_count != 681) return -21;
    if (e->bf16_tensor_count != e->tensor_count) return -22;
    return 0;
}

static char *mu_read_text_file(const char *dir, const char *name) {
    char path[4096];
    if (mu_join_path(path, sizeof(path), dir, name) != 0) return NULL;
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

static int mu_json_any_int(const char *json, const char *key, int expected) {
    char pattern[128];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return 0;
    const char *p = json;
    while ((p = strstr(p, pattern)) != NULL) {
        p += n;
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (*p != ':') continue;
        p++;
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        char *endp = NULL;
        long v = strtol(p, &endp, 10);
        if (endp != p && v == expected) return 1;
        p = endp && endp > p ? endp : p + 1;
    }
    return 0;
}

static int mu_json_any_string(const char *json, const char *key, const char *expected) {
    char pattern[128];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return 0;
    const char *p = json;
    while ((p = strstr(p, pattern)) != NULL) {
        p += n;
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (*p != ':') continue;
        p++;
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (*p != '"') continue;
        const char *start = p + 1;
        const char *after = mu_json_skip_string(p, p + strlen(p));
        if (after <= start) return 0;
        if (mu_json_key_equals(start, after - 1, expected)) return 1;
        p = after;
    }
    return 0;
}

static int mu_validate_config(mu_engine *e) {
    char *config = mu_read_text_file(e->opt.model_dir, "config.json");
    char *preproc = mu_read_text_file(e->opt.model_dir, "preprocessor_config.json");
    if (!config || !preproc) {
        free(config);
        free(preproc);
        return -30;
    }

    int ok = 1;
    ok = ok && mu_json_any_string(config, "model_type", "qwen2_vl");
    ok = ok && mu_json_any_int(config, "num_hidden_layers", 24);
    ok = ok && mu_json_any_int(config, "hidden_size", 896);
    ok = ok && mu_json_any_int(config, "num_attention_heads", 14);
    ok = ok && mu_json_any_int(config, "num_key_value_heads", 2);
    ok = ok && mu_json_any_int(config, "intermediate_size", 4864);
    ok = ok && mu_json_any_int(config, "vocab_size", 151936);
    ok = ok && mu_json_any_int(config, "depth", 32);
    ok = ok && mu_json_any_int(config, "embed_dim", 1280);
    ok = ok && mu_json_any_int(config, "patch_size", 14);
    ok = ok && mu_json_any_int(config, "temporal_patch_size", 2);
    ok = ok && mu_json_any_int(config, "spatial_merge_size", 2);

    ok = ok && mu_json_any_int(preproc, "min_pixels", 50176);
    ok = ok && mu_json_any_int(preproc, "max_pixels", 1605632);
    ok = ok && mu_json_any_int(preproc, "patch_size", 14);
    ok = ok && mu_json_any_int(preproc, "temporal_patch_size", 2);
    ok = ok && mu_json_any_int(preproc, "merge_size", 2);

    free(config);
    free(preproc);
    if (!ok) return -31;

    e->cfg.text_layers = 24;
    e->cfg.hidden_size = 896;
    e->cfg.text_heads = 14;
    e->cfg.kv_heads = 2;
    e->cfg.intermediate_size = 4864;
    e->cfg.vocab_size = 151936;
    e->cfg.vision_layers = 32;
    e->cfg.vision_embed_dim = 1280;
    e->cfg.vision_heads = 16;
    e->cfg.patch_size = 14;
    e->cfg.temporal_patch_size = 2;
    e->cfg.spatial_merge_size = 2;
    return 0;
}

static int mu_utf8_from_codepoint(int cp, char out[5]) {
    if (cp <= 0x7f) {
        out[0] = (char)cp;
        out[1] = 0;
        return 1;
    }
    if (cp <= 0x7ff) {
        out[0] = (char)(0xc0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3f));
        out[2] = 0;
        return 2;
    }
    if (cp <= 0xffff) {
        out[0] = (char)(0xe0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        out[2] = (char)(0x80 | (cp & 0x3f));
        out[3] = 0;
        return 3;
    }
    out[0] = (char)(0xf0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
    out[3] = (char)(0x80 | (cp & 0x3f));
    out[4] = 0;
    return 4;
}

static int mu_init_byte_encoder(mu_engine *e) {
    int used[256] = {0};
    for (int b = 33; b <= 126; b++) used[b] = 1;
    for (int b = 161; b <= 172; b++) used[b] = 1;
    for (int b = 174; b <= 255; b++) used[b] = 1;
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        int cp = used[b] ? b : 256 + extra++;
        char tmp[5];
        int len = mu_utf8_from_codepoint(cp, tmp);
        e->byte_encoder[b] = mu_strndup_c99(tmp, (size_t)len);
        if (!e->byte_encoder[b]) return -1;
    }
    return 0;
}

static int mu_json_unescape_string(const char **pp, char **out) {
    const char *p = *pp;
    if (*p != '"') return -1;
    p++;
    size_t cap = strlen(p) + 1;
    char *buf = (char *)malloc(cap);
    if (!buf) return -2;
    size_t len = 0;
    while (*p) {
        if (*p == '"') {
            buf[len] = 0;
            *pp = p + 1;
            *out = buf;
            return 0;
        }
        if (*p == '\\') {
            p++;
            if (!*p) break;
            if (*p == 'n') buf[len++] = '\n';
            else if (*p == 'r') buf[len++] = '\r';
            else if (*p == 't') buf[len++] = '\t';
            else if (*p == '"' || *p == '\\' || *p == '/') buf[len++] = *p;
            else if (*p == 'u') {
                /* The shipped Qwen2-VL vocab uses UTF-8 keys for byte-level
                 * symbols.  Keep a conservative ASCII escape fallback. */
                int cp = 0;
                for (int i = 0; i < 4 && p[1 + i]; i++) {
                    char c = p[1 + i];
                    cp <<= 4;
                    if (c >= '0' && c <= '9') cp |= c - '0';
                    else if (c >= 'a' && c <= 'f') cp |= c - 'a' + 10;
                    else if (c >= 'A' && c <= 'F') cp |= c - 'A' + 10;
                }
                char tmp[5];
                int n = mu_utf8_from_codepoint(cp, tmp);
                memcpy(buf + len, tmp, (size_t)n);
                len += (size_t)n;
                p += 4;
            } else {
                buf[len++] = *p;
            }
            p++;
            continue;
        }
        buf[len++] = *p++;
    }
    free(buf);
    return -3;
}

static int mu_load_vocab(mu_engine *e) {
    char *json = mu_read_text_file(e->opt.model_dir, "vocab.json");
    if (!json) return -1;
    if (mu_hash_init(&e->vocab, 524288) != 0) {
        free(json);
        return -2;
    }

    const char *p = json;
    p = strchr(p, '{');
    if (!p) {
        free(json);
        return -3;
    }
    p++;
    while (*p) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (*p == '}') break;
        char *key = NULL;
        if (mu_json_unescape_string(&p, &key) != 0) {
            free(json);
            return -4;
        }
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (*p != ':') {
            free(key);
            free(json);
            return -5;
        }
        p++;
        char *endp = NULL;
        long id = strtol(p, &endp, 10);
        if (endp == p) {
            free(key);
            free(json);
            return -6;
        }
        if (mu_hash_put(&e->vocab, key, (int)id) != 0) {
            free(key);
            free(json);
            return -7;
        }
        free(key);
        p = endp;
    }
    free(json);
    return 0;
}

static int mu_load_merges(mu_engine *e) {
    char path[4096];
    if (mu_join_path(path, sizeof(path), e->opt.model_dir, "merges.txt") != 0) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -2;
    if (mu_hash_init(&e->merges, 524288) != 0) {
        fclose(fp);
        return -3;
    }
    char line[8192];
    int rank = 0;
    while (fgets(line, sizeof(line), fp)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        if (!len || line[0] == '#') continue;
        char *space = strchr(line, ' ');
        if (!space) continue;
        *space = 0;
        const char *a = line;
        const char *b = space + 1;
        size_t klen = strlen(a) + strlen(b) + 2;
        char *key = (char *)malloc(klen);
        if (!key) {
            fclose(fp);
            return -4;
        }
        snprintf(key, klen, "%s\t%s", a, b);
        if (mu_hash_put(&e->merges, key, rank++) != 0) {
            free(key);
            fclose(fp);
            return -5;
        }
        free(key);
    }
    fclose(fp);
    return 0;
}

static int mu_ensure_tokenizer(mu_engine *e) {
    if (e->tokenizer_loaded) return 0;
    if (mu_init_byte_encoder(e) != 0) return -1;
    if (mu_load_vocab(e) != 0) return -2;
    if (mu_load_merges(e) != 0) return -3;
    e->tokenizer_loaded = 1;
    return 0;
}

static int mu_bind_required_tensors(mu_engine *e) {
    static const char *text_suffixes[] = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.v_proj.weight",
    };
    static const char *vision_suffixes[] = {
        "attn.proj.bias",
        "attn.proj.weight",
        "attn.qkv.bias",
        "attn.qkv.weight",
        "mlp.fc1.bias",
        "mlp.fc1.weight",
        "mlp.fc2.bias",
        "mlp.fc2.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    };
    static const char *global_tensors[] = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "visual.patch_embed.proj.weight",
        "visual.merger.ln_q.bias",
        "visual.merger.ln_q.weight",
        "visual.merger.mlp.0.bias",
        "visual.merger.mlp.0.weight",
        "visual.merger.mlp.2.bias",
        "visual.merger.mlp.2.weight",
    };

    for (size_t i = 0; i < sizeof(global_tensors) / sizeof(global_tensors[0]); i++) {
        if (!mu_header_has_tensor(&e->st, global_tensors[i])) return -40;
    }

    char name[512];
    for (int layer = 0; layer < e->cfg.text_layers; layer++) {
        for (size_t i = 0; i < sizeof(text_suffixes) / sizeof(text_suffixes[0]); i++) {
            int n = snprintf(name, sizeof(name), "model.layers.%d.%s", layer, text_suffixes[i]);
            if (n < 0 || (size_t)n >= sizeof(name)) return -41;
            if (!mu_header_has_tensor(&e->st, name)) return -42;
        }
        e->bound_text_layers++;
    }

    for (int layer = 0; layer < e->cfg.vision_layers; layer++) {
        for (size_t i = 0; i < sizeof(vision_suffixes) / sizeof(vision_suffixes[0]); i++) {
            int n = snprintf(name, sizeof(name), "visual.blocks.%d.%s", layer, vision_suffixes[i]);
            if (n < 0 || (size_t)n >= sizeof(name)) return -43;
            if (!mu_header_has_tensor(&e->st, name)) return -44;
        }
        e->bound_vision_layers++;
    }
    return 0;
}

static const mu_special_token g_mu_special_tokens[] = {
    {"<|endoftext|>", 151643},
    {"<|im_start|>", 151644},
    {"<|im_end|>", 151645},
    {"<|object_ref_start|>", 151646},
    {"<|object_ref_end|>", 151647},
    {"<|box_start|>", 151648},
    {"<|box_end|>", 151649},
    {"<|quad_start|>", 151650},
    {"<|quad_end|>", 151651},
    {"<|vision_start|>", 151652},
    {"<|vision_end|>", 151653},
    {"<|vision_pad|>", 151654},
    {"<|image_pad|>", 151655},
    {"<|video_pad|>", 151656},
    {"<|ref_start|>", 151657},
    {"<|ref_end|>", 151658},
    {"<|md_start|>", 151659},
    {"<|md_end|>", 151660},
    {"<ched>", 151661},
    {"<ecel>", 151662},
    {"<fcel>", 151663},
    {"<lcel>", 151664},
    {"<ucel>", 151665},
    {"<xcel>", 151666},
    {"<nl>", 151667},
    {"<|rotate_up|>", 151668},
    {"<|rotate_down|>", 151669},
    {"<|rotate_left|>", 151670},
    {"<|rotate_right|>", 151671},
    {"<|txt_contd|>", 151672},
    {"<|paratext|>", 151673},
};

static int mu_match_special(const char *p, int *id, size_t *len) {
    size_t best_len = 0;
    int best_id = -1;
    for (size_t i = 0; i < sizeof(g_mu_special_tokens) / sizeof(g_mu_special_tokens[0]); i++) {
        size_t n = strlen(g_mu_special_tokens[i].text);
        if (n > best_len && strncmp(p, g_mu_special_tokens[i].text, n) == 0) {
            best_len = n;
            best_id = g_mu_special_tokens[i].id;
        }
    }
    if (best_id < 0) return 0;
    *id = best_id;
    *len = best_len;
    return 1;
}

static const char *mu_next_special_start(const char *p, const char *end) {
    const char *best = end;
    for (size_t i = 0; i < sizeof(g_mu_special_tokens) / sizeof(g_mu_special_tokens[0]); i++) {
        const char *q = strstr(p, g_mu_special_tokens[i].text);
        if (q && q < best) best = q;
    }
    return best;
}

static int mu_is_alpha_ascii(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static int mu_is_digit_ascii(unsigned char c) {
    return c >= '0' && c <= '9';
}

static int mu_is_space_ascii(unsigned char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v';
}

static int mu_is_alnum_ascii(unsigned char c) {
    return mu_is_alpha_ascii(c) || mu_is_digit_ascii(c);
}

static const char *mu_next_pretoken(const char *p, const char *end, const char **tok_end) {
    const unsigned char *u = (const unsigned char *)p;
    const unsigned char *ue = (const unsigned char *)end;
    if (u >= ue) {
        *tok_end = p;
        return p;
    }
    if (*u == ' ' && u + 1 < ue && mu_is_alpha_ascii(u[1])) {
        u++;
        while (u < ue && mu_is_alpha_ascii(*u)) u++;
        *tok_end = (const char *)u;
        return p;
    }
    if (mu_is_alpha_ascii(*u)) {
        while (u < ue && mu_is_alpha_ascii(*u)) u++;
        *tok_end = (const char *)u;
        return p;
    }
    if (mu_is_digit_ascii(*u)) {
        *tok_end = (const char *)(u + 1);
        return p;
    }
    if (*u == ' ' && u + 1 < ue && !mu_is_space_ascii(u[1]) && !mu_is_alnum_ascii(u[1])) {
        u++;
        while (u < ue && !mu_is_space_ascii(*u) && !mu_is_alnum_ascii(*u)) u++;
        while (u < ue && (*u == '\n' || *u == '\r')) u++;
        *tok_end = (const char *)u;
        return p;
    }
    if (!mu_is_space_ascii(*u) && !mu_is_alnum_ascii(*u)) {
        while (u < ue && !mu_is_space_ascii(*u) && !mu_is_alnum_ascii(*u)) u++;
        while (u < ue && (*u == '\n' || *u == '\r')) u++;
        *tok_end = (const char *)u;
        return p;
    }
    if (*u == '\n' || *u == '\r') {
        while (u < ue && (*u == '\n' || *u == '\r')) u++;
        *tok_end = (const char *)u;
        return p;
    }
    if (mu_is_space_ascii(*u)) {
        while (u < ue && mu_is_space_ascii(*u) && *u != '\n' && *u != '\r') u++;
        *tok_end = (const char *)u;
        return p;
    }
    /* Fallback: consume one UTF-8 codepoint. */
    if ((*u & 0x80) == 0) u++;
    else if ((*u & 0xe0) == 0xc0 && u + 2 <= ue) u += 2;
    else if ((*u & 0xf0) == 0xe0 && u + 3 <= ue) u += 3;
    else if ((*u & 0xf8) == 0xf0 && u + 4 <= ue) u += 4;
    else u++;
    *tok_end = (const char *)u;
    return p;
}

static char *mu_pair_key(const char *a, const char *b) {
    size_t len = strlen(a) + strlen(b) + 2;
    char *key = (char *)malloc(len);
    if (!key) return NULL;
    snprintf(key, len, "%s\t%s", a, b);
    return key;
}

static int mu_bpe_piece(mu_engine *e, const char *start, const char *end, int *out, int max_out) {
    int nbytes = (int)(end - start);
    if (nbytes <= 0) return 0;
    char **symbols = (char **)calloc((size_t)nbytes, sizeof(symbols[0]));
    if (!symbols) return -1;
    int nsym = 0;
    for (const unsigned char *p = (const unsigned char *)start; p < (const unsigned char *)end; p++) {
        symbols[nsym] = mu_strdup_c99(e->byte_encoder[*p]);
        if (!symbols[nsym]) {
            for (int i = 0; i < nsym; i++) free(symbols[i]);
            free(symbols);
            return -2;
        }
        nsym++;
    }

    while (nsym > 1) {
        int best = -1;
        int best_rank = 0x7fffffff;
        for (int i = 0; i + 1 < nsym; i++) {
            char *key = mu_pair_key(symbols[i], symbols[i + 1]);
            if (!key) continue;
            int rank = 0;
            int found = mu_hash_get(&e->merges, key, &rank);
            free(key);
            if (found && rank < best_rank) {
                best_rank = rank;
                best = i;
            }
        }
        if (best < 0) break;
        size_t merged_len = strlen(symbols[best]) + strlen(symbols[best + 1]) + 1;
        char *merged = (char *)malloc(merged_len);
        if (!merged) break;
        snprintf(merged, merged_len, "%s%s", symbols[best], symbols[best + 1]);
        free(symbols[best]);
        free(symbols[best + 1]);
        symbols[best] = merged;
        for (int i = best + 1; i + 1 < nsym; i++) symbols[i] = symbols[i + 1];
        nsym--;
    }

    int nout = 0;
    for (int i = 0; i < nsym; i++) {
        int id = -1;
        if (!mu_hash_get(&e->vocab, symbols[i], &id)) {
            nout = -10;
            break;
        }
        if (nout >= max_out) {
            nout = -11;
            break;
        }
        out[nout++] = id;
    }
    for (int i = 0; i < nsym; i++) free(symbols[i]);
    free(symbols);
    return nout;
}

int mu_tokenize_text(mu_engine *e, const char *text, int *out, int max_out) {
    if (!e || !text || !out || max_out <= 0) return -1;
    int rc = mu_ensure_tokenizer(e);
    if (rc) return -2;
    int nout = 0;
    const char *p = text;
    const char *end = text + strlen(text);
    while (p < end) {
        int sid = 0;
        size_t slen = 0;
        if (mu_match_special(p, &sid, &slen)) {
            if (nout >= max_out) return -3;
            out[nout++] = sid;
            p += slen;
            continue;
        }
        const char *segment_end = mu_next_special_start(p, end);
        const char *tok_end = p;
        mu_next_pretoken(p, segment_end, &tok_end);
        if (tok_end <= p) return -4;
        int got = mu_bpe_piece(e, p, tok_end, out + nout, max_out - nout);
        if (got < 0) return got;
        nout += got;
        p = tok_end;
    }
    return nout;
}

int mu_tokenize_image_text(mu_engine *e, const char *text,
                           int grid_t, int grid_h, int grid_w,
                           int *out, int max_out) {
    if (!e || !text || !out || max_out <= 0) return -1;
    int merge = e->cfg.spatial_merge_size > 0 ? e->cfg.spatial_merge_size : 2;
    long image_tokens = (long)grid_t * (long)grid_h * (long)grid_w;
    long denom = (long)merge * (long)merge;
    if (grid_t <= 0 || grid_h <= 0 || grid_w <= 0 || image_tokens <= 0) return -2;
    if (image_tokens % denom != 0) return -3;
    image_tokens /= denom;
    if (image_tokens <= 0 || image_tokens > max_out) return -4;

    int *base = (int *)malloc((size_t)max_out * sizeof(base[0]));
    if (!base) return -5;
    int base_n = mu_tokenize_text(e, text, base, max_out);
    if (base_n < 0) {
        free(base);
        return base_n;
    }

    int nout = 0;
    for (int i = 0; i < base_n; i++) {
        int repeat = base[i] == 151655 ? (int)image_tokens : 1;
        if (nout + repeat > max_out) {
            free(base);
            return -6;
        }
        for (int j = 0; j < repeat; j++) out[nout++] = base[i];
    }
    free(base);
    return nout;
}

int mu_build_position_ids(mu_engine *e, const int *input_ids, int n_ids,
                          int grid_t, int grid_h, int grid_w,
                          int *out, int max_out) {
    if (!e || !input_ids || n_ids <= 0 || !out || max_out < n_ids * 3) return -1;

    if (grid_t <= 0 || grid_h <= 0 || grid_w <= 0) {
        for (int row = 0; row < 3; row++) {
            for (int i = 0; i < n_ids; i++) out[row * n_ids + i] = i;
        }
        return n_ids * 3;
    }

    int merge = e->cfg.spatial_merge_size > 0 ? e->cfg.spatial_merge_size : 2;
    if (grid_h % merge != 0 || grid_w % merge != 0) return -2;
    int llm_t = grid_t;
    int llm_h = grid_h / merge;
    int llm_w = grid_w / merge;
    long image_tokens = (long)llm_t * (long)llm_h * (long)llm_w;
    if (llm_t <= 0 || llm_h <= 0 || llm_w <= 0 || image_tokens <= 0 || image_tokens > n_ids) return -3;

    int image_start = -1;
    for (int i = 0; i < n_ids; i++) {
        if (input_ids[i] == 151655) {
            image_start = i;
            break;
        }
    }
    if (image_start < 0 || image_start + image_tokens > n_ids) return -4;
    for (long i = 0; i < image_tokens; i++) {
        if (input_ids[image_start + (int)i] != 151655) return -5;
    }

    for (int i = 0; i < image_start; i++) {
        out[0 * n_ids + i] = i;
        out[1 * n_ids + i] = i;
        out[2 * n_ids + i] = i;
    }

    for (int t = 0; t < llm_t; t++) {
        for (int h = 0; h < llm_h; h++) {
            for (int w = 0; w < llm_w; w++) {
                int idx = image_start + (t * llm_h + h) * llm_w + w;
                out[0 * n_ids + idx] = image_start + t;
                out[1 * n_ids + idx] = image_start + h;
                out[2 * n_ids + idx] = image_start + w;
            }
        }
    }

    int image_end = image_start + (int)image_tokens;
    int max_axis = llm_t - 1;
    if (llm_h - 1 > max_axis) max_axis = llm_h - 1;
    if (llm_w - 1 > max_axis) max_axis = llm_w - 1;
    int text_start = image_start + max_axis + 1;
    for (int i = image_end; i < n_ids; i++) {
        int pos = text_start + (i - image_end);
        out[0 * n_ids + i] = pos;
        out[1 * n_ids + i] = pos;
        out[2 * n_ids + i] = pos;
    }
    return n_ids * 3;
}

static float mu_qwen2vl_norm_bf16(unsigned char v, int channel) {
    static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    float x = ((float)v) * (1.0f / 255.0f);
    float y = (x - mean[channel]) / std[channel];
    return mu_bf16_to_f32(mu_f32_to_bf16(y));
}

static int mu_load_layout_rgb_file_region(const char *path, const float *bbox,
                                          unsigned char **out_rgb, int *out_w, int *out_h) {
#ifdef __APPLE__
    if (!path || !out_rgb || !out_w || !out_h) return -1;
    *out_rgb = NULL;
    *out_w = 0;
    *out_h = 0;

    const int dst_w = 1036;
    const int dst_h = 1036;
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        NULL, (const UInt8 *)path, (CFIndex)strlen(path), false);
    if (!url) return -2;
    CGImageSourceRef src = CGImageSourceCreateWithURL(url, NULL);
    if (!src) {
        CFRelease(url);
        return -3;
    }
    CGImageRef image = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    if (!image) {
        CFRelease(src);
        CFRelease(url);
        return -4;
    }

    size_t rgba_stride = (size_t)dst_w * 4u;
    unsigned char *rgba = (unsigned char *)calloc((size_t)dst_h, rgba_stride);
    unsigned char *rgb = (unsigned char *)malloc((size_t)dst_w * (size_t)dst_h * 3u);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    if (!rgba || !rgb || !cs) {
        if (cs) CGColorSpaceRelease(cs);
        free(rgba);
        free(rgb);
        CGImageRelease(image);
        CFRelease(src);
        CFRelease(url);
        return -5;
    }
    CGContextRef ctx = CGBitmapContextCreate(
        rgba, (size_t)dst_w, (size_t)dst_h, 8, rgba_stride, cs,
        kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast);
    if (!ctx) {
        CGColorSpaceRelease(cs);
        free(rgba);
        free(rgb);
        CGImageRelease(image);
        CFRelease(src);
        CFRelease(url);
        return -6;
    }
    CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
    CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0);
    CGContextFillRect(ctx, CGRectMake(0.0, 0.0, (CGFloat)dst_w, (CGFloat)dst_h));
    if (bbox) {
        size_t src_w = CGImageGetWidth(image);
        size_t src_h = CGImageGetHeight(image);
        float x1 = bbox[0], y1 = bbox[1], x2 = bbox[2], y2 = bbox[3];
        if (x1 < 0.0f) x1 = 0.0f;
        if (y1 < 0.0f) y1 = 0.0f;
        if (x2 > 1.0f) x2 = 1.0f;
        if (y2 > 1.0f) y2 = 1.0f;
        if (x2 <= x1 || y2 <= y1) {
            CGContextRelease(ctx);
            CGColorSpaceRelease(cs);
            free(rgba);
            free(rgb);
            CGImageRelease(image);
            CFRelease(src);
            CFRelease(url);
            return -7;
        }
        CGFloat crop_x = (CGFloat)x1 * (CGFloat)src_w;
        CGFloat crop_y = (CGFloat)y1 * (CGFloat)src_h;
        CGFloat crop_w = (CGFloat)(x2 - x1) * (CGFloat)src_w;
        CGFloat crop_h = (CGFloat)(y2 - y1) * (CGFloat)src_h;
        CGFloat sx = (CGFloat)dst_w / crop_w;
        CGFloat sy = (CGFloat)dst_h / crop_h;
        CGContextDrawImage(ctx,
                           CGRectMake(-crop_x * sx, -crop_y * sy,
                                      (CGFloat)src_w * sx, (CGFloat)src_h * sy),
                           image);
    } else {
        CGContextDrawImage(ctx, CGRectMake(0.0, 0.0, (CGFloat)dst_w, (CGFloat)dst_h), image);
    }

    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            const unsigned char *p = rgba + (size_t)y * rgba_stride + (size_t)x * 4u;
            unsigned char *q = rgb + ((size_t)y * (size_t)dst_w + (size_t)x) * 3u;
            q[0] = p[0];
            q[1] = p[1];
            q[2] = p[2];
        }
    }

    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CGImageRelease(image);
    CFRelease(src);
    CFRelease(url);
    free(rgba);
    *out_rgb = rgb;
    *out_w = dst_w;
    *out_h = dst_h;
    return 0;
#else
    (void)path;
    (void)out_rgb;
    (void)out_w;
    (void)out_h;
    return -200;
#endif
}

static int mu_load_layout_rgb_file(const char *path, unsigned char **out_rgb, int *out_w, int *out_h) {
    return mu_load_layout_rgb_file_region(path, NULL, out_rgb, out_w, out_h);
}

static int mu_qwen2vl_smart_resize(int height, int width, int *out_h, int *out_w) {
    const int factor = 28;
    const int min_pixels = 50176;
    const int max_pixels = 1605632;
    if (height <= 0 || width <= 0 || !out_h || !out_w) return -1;
    double ratio = (double)(height > width ? height : width) /
                   (double)(height < width ? height : width);
    if (ratio > 200.0) return -2;

    int h_bar = (int)(round((double)height / (double)factor) * (double)factor);
    int w_bar = (int)(round((double)width / (double)factor) * (double)factor);
    if (h_bar < factor) h_bar = factor;
    if (w_bar < factor) w_bar = factor;

    long pixels = (long)h_bar * (long)w_bar;
    if (pixels > max_pixels) {
        double beta = sqrt(((double)height * (double)width) / (double)max_pixels);
        h_bar = (int)(floor((double)height / beta / (double)factor) * (double)factor);
        w_bar = (int)(floor((double)width / beta / (double)factor) * (double)factor);
        if (h_bar < factor) h_bar = factor;
        if (w_bar < factor) w_bar = factor;
    } else if (pixels < min_pixels) {
        double beta = sqrt((double)min_pixels / ((double)height * (double)width));
        h_bar = (int)(ceil((double)height * beta / (double)factor) * (double)factor);
        w_bar = (int)(ceil((double)width * beta / (double)factor) * (double)factor);
    }

    *out_h = h_bar;
    *out_w = w_bar;
    return 0;
}

static int mu_load_extract_rgb_file_region(const char *path, const float bbox[4],
                                           unsigned char **out_rgb, int *out_w, int *out_h) {
#ifdef __APPLE__
    if (!path || !bbox || !out_rgb || !out_w || !out_h) return -1;
    *out_rgb = NULL;
    *out_w = 0;
    *out_h = 0;

    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        NULL, (const UInt8 *)path, (CFIndex)strlen(path), false);
    if (!url) return -2;
    CGImageSourceRef src = CGImageSourceCreateWithURL(url, NULL);
    if (!src) {
        CFRelease(url);
        return -3;
    }
    CGImageRef image = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    if (!image) {
        CFRelease(src);
        CFRelease(url);
        return -4;
    }

    size_t src_w = CGImageGetWidth(image);
    size_t src_h = CGImageGetHeight(image);
    float x1 = bbox[0], y1 = bbox[1], x2 = bbox[2], y2 = bbox[3];
    if (x1 < 0.0f) x1 = 0.0f;
    if (y1 < 0.0f) y1 = 0.0f;
    if (x2 > 1.0f) x2 = 1.0f;
    if (y2 > 1.0f) y2 = 1.0f;
    if (x2 <= x1 || y2 <= y1) {
        CGImageRelease(image);
        CFRelease(src);
        CFRelease(url);
        return -5;
    }

    int crop_l = (int)lrint((double)x1 * (double)src_w);
    int crop_t = (int)lrint((double)y1 * (double)src_h);
    int crop_r = (int)lrint((double)x2 * (double)src_w);
    int crop_b = (int)lrint((double)y2 * (double)src_h);
    if (crop_l < 0) crop_l = 0;
    if (crop_t < 0) crop_t = 0;
    if (crop_r > (int)src_w) crop_r = (int)src_w;
    if (crop_b > (int)src_h) crop_b = (int)src_h;
    if (crop_r <= crop_l) crop_r = crop_l + 1;
    if (crop_b <= crop_t) crop_b = crop_t + 1;
    if (crop_r > (int)src_w || crop_b > (int)src_h) {
        CGImageRelease(image);
        CFRelease(src);
        CFRelease(url);
        return -6;
    }

    int crop_w = crop_r - crop_l;
    int crop_h = crop_b - crop_t;
    int pad_w = crop_w;
    int pad_h = crop_h;
    double edge_ratio = (double)(pad_w > pad_h ? pad_w : pad_h) /
                        (double)(pad_w < pad_h ? pad_w : pad_h);
    if (edge_ratio > 50.0) {
        if (pad_w > pad_h) pad_h = (int)ceil((double)pad_w / 50.0);
        else pad_w = (int)ceil((double)pad_h / 50.0);
    }

    int prep_w = pad_w;
    int prep_h = pad_h;
    int min_edge = prep_w < prep_h ? prep_w : prep_h;
    if (min_edge < 28) {
        double scale = 28.0 / (double)min_edge;
        prep_w = (int)ceil((double)prep_w * scale);
        prep_h = (int)ceil((double)prep_h * scale);
    }

    int dst_w = 0;
    int dst_h = 0;
    if (mu_qwen2vl_smart_resize(prep_h, prep_w, &dst_h, &dst_w) != 0) {
        CGImageRelease(image);
        CFRelease(src);
        CFRelease(url);
        return -7;
    }

    size_t rgba_stride = (size_t)dst_w * 4u;
    unsigned char *rgba = (unsigned char *)calloc((size_t)dst_h, rgba_stride);
    unsigned char *rgb = (unsigned char *)malloc((size_t)dst_w * (size_t)dst_h * 3u);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    if (!rgba || !rgb || !cs) {
        if (cs) CGColorSpaceRelease(cs);
        free(rgba);
        free(rgb);
        CGImageRelease(image);
        CFRelease(src);
        CFRelease(url);
        return -8;
    }
    CGContextRef ctx = CGBitmapContextCreate(
        rgba, (size_t)dst_w, (size_t)dst_h, 8, rgba_stride, cs,
        kCGBitmapByteOrder32Big | kCGImageAlphaPremultipliedLast);
    if (!ctx) {
        CGColorSpaceRelease(cs);
        free(rgba);
        free(rgb);
        CGImageRelease(image);
        CFRelease(src);
        CFRelease(url);
        return -9;
    }

    CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
    CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0);
    CGContextFillRect(ctx, CGRectMake(0.0, 0.0, (CGFloat)dst_w, (CGFloat)dst_h));

    double sx_pre = (double)prep_w / (double)pad_w;
    double sy_pre = (double)prep_h / (double)pad_h;
    double sx_final = (double)dst_w / (double)prep_w;
    double sy_final = (double)dst_h / (double)prep_h;
    double paste_x = ((double)pad_w - (double)crop_w) * 0.5;
    double paste_y = ((double)pad_h - (double)crop_h) * 0.5;
    double dest_x = paste_x * sx_pre * sx_final;
    double dest_y = paste_y * sy_pre * sy_final;
    double dest_w = (double)crop_w * sx_pre * sx_final;
    double dest_h = (double)crop_h * sy_pre * sy_final;
    double sx = dest_w / (double)crop_w;
    double sy = dest_h / (double)crop_h;
    CGContextDrawImage(ctx,
                       CGRectMake((CGFloat)(dest_x - (double)crop_l * sx),
                                  (CGFloat)(dest_y - ((double)src_h - (double)crop_b) * sy),
                                  (CGFloat)((double)src_w * sx),
                                  (CGFloat)((double)src_h * sy)),
                       image);

    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            const unsigned char *p = rgba + (size_t)y * rgba_stride + (size_t)x * 4u;
            unsigned char *q = rgb + ((size_t)y * (size_t)dst_w + (size_t)x) * 3u;
            q[0] = p[0];
            q[1] = p[1];
            q[2] = p[2];
        }
    }

    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CGImageRelease(image);
    CFRelease(src);
    CFRelease(url);
    free(rgba);
    mu_debug_dump_rgb("extract", rgb, dst_w, dst_h);
    *out_rgb = rgb;
    *out_w = dst_w;
    *out_h = dst_h;
    return 0;
#else
    (void)path;
    (void)bbox;
    (void)out_rgb;
    (void)out_w;
    (void)out_h;
    return -200;
#endif
}

int mu_preprocess_layout_image_file(mu_engine *e, const char *path, mu_image_tokens *out) {
    if (!e || !path || !out) return -1;
    memset(out, 0, sizeof(*out));
    unsigned char *rgb = NULL;
    int width = 0;
    int height = 0;
    int rc = mu_load_layout_rgb_file(path, &rgb, &width, &height);
    if (rc) return rc;

    const int patch = 14;
    const int temporal = 2;
    const int merge = e->cfg.spatial_merge_size > 0 ? e->cfg.spatial_merge_size : 2;
    if (width != 1036 || height != 1036 || width % patch != 0 || height % patch != 0) {
        free(rgb);
        return -10;
    }
    int grid_t = 1;
    int grid_h = height / patch;
    int grid_w = width / patch;
    if (grid_h % merge != 0 || grid_w % merge != 0) {
        free(rgb);
        return -11;
    }
    int rows = grid_t * grid_h * grid_w;
    int cols = 3 * temporal * patch * patch;
    float *values = (float *)malloc((size_t)rows * (size_t)cols * sizeof(values[0]));
    if (!values) {
        free(rgb);
        return -12;
    }

    int group_h = grid_h / merge;
    int group_w = grid_w / merge;
    for (int gt = 0; gt < grid_t; gt++) {
        for (int ghg = 0; ghg < group_h; ghg++) {
            for (int gwg = 0; gwg < group_w; gwg++) {
                for (int mh = 0; mh < merge; mh++) {
                    for (int mw = 0; mw < merge; mw++) {
                        int row = (((gt * group_h + ghg) * group_w + gwg) * merge + mh) * merge + mw;
                        int patch_y = (ghg * merge + mh) * patch;
                        int patch_x = (gwg * merge + mw) * patch;
                        int col = 0;
                        for (int c = 0; c < 3; c++) {
                            for (int tp = 0; tp < temporal; tp++) {
                                (void)tp;
                                for (int py = 0; py < patch; py++) {
                                    for (int px = 0; px < patch; px++) {
                                        const unsigned char *p =
                                            rgb + ((size_t)(patch_y + py) * (size_t)width + (size_t)(patch_x + px)) * 3u;
                                        values[(size_t)row * (size_t)cols + (size_t)col++] =
                                            mu_qwen2vl_norm_bf16(p[c], c);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(rgb);
    out->grid_t = grid_t;
    out->grid_h = grid_h;
    out->grid_w = grid_w;
    out->rows = rows;
    out->cols = cols;
    out->values = values;
    return 0;
}

static int mu_preprocess_layout_image_region_file(mu_engine *e, const char *path,
                                                  const float bbox[4],
                                                  mu_image_tokens *out) {
    if (!e || !path || !bbox || !out) return -1;
    memset(out, 0, sizeof(*out));
    unsigned char *rgb = NULL;
    int width = 0;
    int height = 0;
    int rc = mu_load_extract_rgb_file_region(path, bbox, &rgb, &width, &height);
    if (rc) return rc;

    const int patch = 14;
    const int temporal = 2;
    const int merge = e->cfg.spatial_merge_size > 0 ? e->cfg.spatial_merge_size : 2;
    if (width <= 0 || height <= 0 || width % patch != 0 || height % patch != 0) {
        free(rgb);
        return -10;
    }
    int grid_t = 1;
    int grid_h = height / patch;
    int grid_w = width / patch;
    if (grid_h % merge != 0 || grid_w % merge != 0) {
        free(rgb);
        return -11;
    }
    int rows = grid_t * grid_h * grid_w;
    int cols = 3 * temporal * patch * patch;
    float *values = (float *)malloc((size_t)rows * (size_t)cols * sizeof(values[0]));
    if (!values) {
        free(rgb);
        return -12;
    }

    int group_h = grid_h / merge;
    int group_w = grid_w / merge;
    for (int gt = 0; gt < grid_t; gt++) {
        for (int ghg = 0; ghg < group_h; ghg++) {
            for (int gwg = 0; gwg < group_w; gwg++) {
                for (int mh = 0; mh < merge; mh++) {
                    for (int mw = 0; mw < merge; mw++) {
                        int row = (((gt * group_h + ghg) * group_w + gwg) * merge + mh) * merge + mw;
                        int patch_y = (ghg * merge + mh) * patch;
                        int patch_x = (gwg * merge + mw) * patch;
                        int col = 0;
                        for (int c = 0; c < 3; c++) {
                            for (int tp = 0; tp < temporal; tp++) {
                                (void)tp;
                                for (int py = 0; py < patch; py++) {
                                    for (int px = 0; px < patch; px++) {
                                        const unsigned char *p =
                                            rgb + ((size_t)(patch_y + py) * (size_t)width + (size_t)(patch_x + px)) * 3u;
                                        values[(size_t)row * (size_t)cols + (size_t)col++] =
                                            mu_qwen2vl_norm_bf16(p[c], c);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    free(rgb);
    out->grid_t = grid_t;
    out->grid_h = grid_h;
    out->grid_w = grid_w;
    out->rows = rows;
    out->cols = cols;
    out->values = values;
    return 0;
}

void mu_image_tokens_free(mu_image_tokens *tokens) {
    if (!tokens) return;
    free(tokens->values);
    memset(tokens, 0, sizeof(*tokens));
}

static const mu_tensor *mu_tensor_by_name(const mu_engine *e, const char *name) {
    if (!e || !name) return NULL;
    int idx = mu_engine_tensor_index(e, name);
    return idx >= 0 ? &e->tensors[idx] : NULL;
}

static const uint16_t *mu_tensor_bf16(const mu_engine *e, const char *name,
                                      int ndim, uint64_t d0, uint64_t d1) {
    const mu_tensor *t = mu_tensor_by_name(e, name);
    if (!t || t->ndim != ndim) return NULL;
    if (ndim >= 1 && t->shape[0] != d0) return NULL;
    if (ndim >= 2 && t->shape[1] != d1) return NULL;
    return (const uint16_t *)t->data;
}

static int mu_bf16_to_f32_array(const uint16_t *src, size_t n, float *dst) {
    if (!src || !dst) return -1;
    for (size_t i = 0; i < n; i++) dst[i] = mu_bf16_to_f32(src[i]);
    return 0;
}

static int mu_linear_seq_f32(const float *x, int seq, int in,
                             const uint16_t *w_bf16, const uint16_t *bias_bf16,
                             int out, float *y, float *w_tmp) {
    if (!x || !w_bf16 || !y || !w_tmp || seq <= 0 || in <= 0 || out <= 0) return -1;
    mu_bf16_to_f32_array(w_bf16, (size_t)out * (size_t)in, w_tmp);
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq, out, in, 1.0f, x, in, w_tmp, in, 0.0f, y, out);
#else
    for (int s = 0; s < seq; s++) {
        for (int r = 0; r < out; r++) {
            float sum = 0.0f;
            const float *wrow = w_tmp + (size_t)r * (size_t)in;
            for (int c = 0; c < in; c++) sum += x[(size_t)s * (size_t)in + (size_t)c] * wrow[c];
            y[(size_t)s * (size_t)out + (size_t)r] = sum;
        }
    }
#endif
    if (bias_bf16) {
        for (int r = 0; r < out; r++) {
            float b = mu_bf16_to_f32(bias_bf16[r]);
            for (int s = 0; s < seq; s++) y[(size_t)s * (size_t)out + (size_t)r] += b;
        }
    }
    return 0;
}

int mu_vision_patch_embed(mu_engine *e, const mu_image_tokens *tokens,
                          float *out, int out_rows, int out_cols) {
    if (!e || !tokens || !tokens->values || !out) return -1;
    const int embed_dim = 1280;
    const int in_dim = 3 * 2 * 14 * 14;
    if (tokens->rows <= 0 || tokens->cols != in_dim ||
        out_rows != tokens->rows || out_cols != embed_dim) {
        return -2;
    }
    const mu_tensor *tw = mu_tensor_by_name(e, "visual.patch_embed.proj.weight");
    if (!tw || tw->ndim != 5 ||
        tw->shape[0] != embed_dim || tw->shape[1] != 3 ||
        tw->shape[2] != 2 || tw->shape[3] != 14 || tw->shape[4] != 14) {
        return -3;
    }
    float *w_tmp = (float *)malloc((size_t)embed_dim * (size_t)in_dim * sizeof(w_tmp[0]));
    if (!w_tmp) return -4;
    int rc = mu_linear_seq_f32(tokens->values, tokens->rows, in_dim,
                               (const uint16_t *)tw->data, NULL,
                               embed_dim, out, w_tmp);
    free(w_tmp);
    if (rc) return -5;
    for (size_t i = 0; i < (size_t)out_rows * (size_t)out_cols; i++) {
        out[i] = mu_bf16_to_f32(mu_f32_to_bf16(out[i]));
    }
    return 0;
}

int mu_vision_rotary_pos_emb(mu_engine *e, int grid_t, int grid_h, int grid_w,
                             float *out, int out_rows, int out_cols) {
    if (!e || !out) return -1;
    const int merge = e->cfg.spatial_merge_size > 0 ? e->cfg.spatial_merge_size : 2;
    const int rotary_cols = 40;
    if (grid_t <= 0 || grid_h <= 0 || grid_w <= 0 ||
        grid_h % merge != 0 || grid_w % merge != 0 ||
        out_rows != grid_t * grid_h * grid_w || out_cols != rotary_cols) {
        return -2;
    }

    float inv_freq[20];
    const float theta = 10000.0f;
    for (int i = 0; i < 20; i++) {
        inv_freq[i] = powf(theta, -((float)(2 * i) / 40.0f));
    }

    int group_h = grid_h / merge;
    int group_w = grid_w / merge;
    for (int t = 0; t < grid_t; t++) {
        for (int ghg = 0; ghg < group_h; ghg++) {
            for (int gwg = 0; gwg < group_w; gwg++) {
                for (int mh = 0; mh < merge; mh++) {
                    for (int mw = 0; mw < merge; mw++) {
                        int row = (((t * group_h + ghg) * group_w + gwg) * merge + mh) * merge + mw;
                        int hpos = ghg * merge + mh;
                        int wpos = gwg * merge + mw;
                        float *dst = out + (size_t)row * rotary_cols;
                        for (int i = 0; i < 20; i++) {
                            dst[i] = (float)hpos * inv_freq[i];
                            dst[20 + i] = (float)wpos * inv_freq[i];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

static int mu_layernorm_one_bf16(const float *x, int n,
                                 const uint16_t *weight, const uint16_t *bias,
                                 float eps, float *out) {
    if (!x || !weight || !bias || !out || n <= 0) return -1;
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv = 1.0f / sqrtf(var / (float)n + eps);
    for (int i = 0; i < n; i++) {
        float y = (x[i] - mean) * inv;
        y = y * mu_bf16_to_f32(weight[i]) + mu_bf16_to_f32(bias[i]);
        out[i] = mu_bf16_to_f32(mu_f32_to_bf16(y));
    }
    return 0;
}

static int mu_linear_one_bf16(const float *x, int in,
                              const uint16_t *w_bf16, const uint16_t *bias_bf16,
                              int out_n, float *out) {
    if (!x || !w_bf16 || !out || in <= 0 || out_n <= 0) return -1;
    for (int r = 0; r < out_n; r++) {
        const uint16_t *wrow = w_bf16 + (size_t)r * (size_t)in;
        float sum = bias_bf16 ? mu_bf16_to_f32(bias_bf16[r]) : 0.0f;
        for (int c = 0; c < in; c++) sum += x[c] * mu_bf16_to_f32(wrow[c]);
        out[r] = mu_bf16_to_f32(mu_f32_to_bf16(sum));
    }
    return 0;
}

int mu_vision_block0_norm1_token0(mu_engine *e, const float *patch_embeds,
                                  int rows, int cols, float *out, int out_n) {
    if (!e || !patch_embeds || !out || rows <= 0 || cols != 1280 || out_n != 1280) {
        return -1;
    }
    const uint16_t *w = mu_tensor_bf16(e, "visual.blocks.0.norm1.weight", 1, 1280, 0);
    const uint16_t *b = mu_tensor_bf16(e, "visual.blocks.0.norm1.bias", 1, 1280, 0);
    if (!w || !b) return -2;
    return mu_layernorm_one_bf16(patch_embeds, 1280, w, b, 1e-6f, out);
}

int mu_vision_block0_qkv_token0(mu_engine *e, const float *norm1,
                                int norm1_n, float *out, int out_n) {
    if (!e || !norm1 || !out || norm1_n != 1280 || out_n != 3840) return -1;
    const uint16_t *w = mu_tensor_bf16(e, "visual.blocks.0.attn.qkv.weight", 2, 3840, 1280);
    const uint16_t *b = mu_tensor_bf16(e, "visual.blocks.0.attn.qkv.bias", 1, 3840, 0);
    if (!w || !b) return -2;
    return mu_linear_one_bf16(norm1, 1280, w, b, 3840, out);
}

static void mu_vision_rope_head(const float *src, const float *rope, float *dst) {
    float old[80];
    memcpy(old, src, sizeof(old));
    for (int d = 0; d < 80; d++) {
        float angle = rope[d < 40 ? d : d - 40];
        float c = cosf(angle);
        float s = sinf(angle);
        float rot = d < 40 ? -old[d + 40] : old[d - 40];
        dst[d] = mu_bf16_to_f32(mu_f32_to_bf16(old[d] * c + rot * s));
    }
}

static int mu_vision_attention_all_f32(const float *qkv, int rows,
                                       const float *rotary, float *out) {
    if (!qkv || !rotary || !out || rows <= 0) return -1;
    const int heads = 16;
    const int head_dim = 80;
    const float scale = 1.0f / sqrtf(80.0f);

    float *qh = (float *)malloc((size_t)rows * head_dim * sizeof(qh[0]));
    float *kh = (float *)malloc((size_t)rows * head_dim * sizeof(kh[0]));
    float *vh = (float *)malloc((size_t)rows * head_dim * sizeof(vh[0]));
    float *oh = (float *)malloc((size_t)rows * head_dim * sizeof(oh[0]));
    float *scores = (float *)malloc((size_t)rows * (size_t)rows * sizeof(scores[0]));
    if (!qh || !kh || !vh || !oh || !scores) {
        free(qh);
        free(kh);
        free(vh);
        free(oh);
        free(scores);
        return -2;
    }

    for (int h = 0; h < heads; h++) {
        for (int r = 0; r < rows; r++) {
            mu_vision_rope_head(qkv + (size_t)r * 3840u + h * head_dim,
                                rotary + (size_t)r * 40u,
                                qh + (size_t)r * head_dim);
            mu_vision_rope_head(qkv + (size_t)r * 3840u + 1280u + h * head_dim,
                                rotary + (size_t)r * 40u,
                                kh + (size_t)r * head_dim);
            memcpy(vh + (size_t)r * head_dim,
                   qkv + (size_t)r * 3840u + 2560u + h * head_dim,
                   (size_t)head_dim * sizeof(vh[0]));
        }

#ifdef __APPLE__
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, rows, head_dim,
                    scale, qh, head_dim, kh, head_dim,
                    0.0f, scores, rows);
#else
        for (int t = 0; t < rows; t++) {
            for (int r = 0; r < rows; r++) {
                float dot = 0.0f;
                const float *qrow = qh + (size_t)t * head_dim;
                const float *krow = kh + (size_t)r * head_dim;
                for (int d = 0; d < head_dim; d++) dot += qrow[d] * krow[d];
                scores[(size_t)t * rows + r] = dot * scale;
            }
        }
#endif

        for (int t = 0; t < rows; t++) {
            float *row = scores + (size_t)t * rows;
            float max_score = -FLT_MAX;
            for (int r = 0; r < rows; r++) {
                if (row[r] > max_score) max_score = row[r];
            }
            float denom = 0.0f;
            for (int r = 0; r < rows; r++) {
                row[r] = expf(row[r] - max_score);
                denom += row[r];
            }
            for (int r = 0; r < rows; r++) {
                row[r] = mu_bf16_to_f32(mu_f32_to_bf16(row[r] / denom));
            }
        }

#ifdef __APPLE__
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows, head_dim, rows,
                    1.0f, scores, rows, vh, head_dim,
                    0.0f, oh, head_dim);
#else
        for (int t = 0; t < rows; t++) {
            float *orow = oh + (size_t)t * head_dim;
            for (int d = 0; d < head_dim; d++) orow[d] = 0.0f;
            for (int r = 0; r < rows; r++) {
                float p = scores[(size_t)t * rows + r];
                const float *vrow = vh + (size_t)r * head_dim;
                for (int d = 0; d < head_dim; d++) orow[d] += p * vrow[d];
            }
        }
#endif

        for (int r = 0; r < rows; r++) {
            float *dst = out + (size_t)r * 1280u + h * head_dim;
            const float *src = oh + (size_t)r * head_dim;
            for (int d = 0; d < head_dim; d++) {
                dst[d] = mu_bf16_to_f32(mu_f32_to_bf16(src[d]));
            }
        }
    }

    free(qh);
    free(kh);
    free(vh);
    free(oh);
    free(scores);
    return 0;
}

static int mu_vision_block0_attn_token(mu_engine *e, const float *patch_embeds,
                                       int rows, int cols,
                                       const float *rotary, int rotary_rows, int rotary_cols,
                                       int token_index, float *out, int out_n);

int mu_vision_block0_attn_token0(mu_engine *e, const float *patch_embeds,
                                 int rows, int cols,
                                 const float *rotary, int rotary_rows, int rotary_cols,
                                 float *out, int out_n) {
    return mu_vision_block0_attn_token(e, patch_embeds, rows, cols,
                                       rotary, rotary_rows, rotary_cols,
                                       0, out, out_n);
}

static int mu_vision_block0_attn_token(mu_engine *e, const float *patch_embeds,
                                       int rows, int cols,
                                       const float *rotary, int rotary_rows, int rotary_cols,
                                       int token_index, float *out, int out_n) {
    if (!e || !patch_embeds || !rotary || !out ||
        rows <= 0 || cols != 1280 ||
        rotary_rows != rows || rotary_cols != 40 ||
        token_index < 0 || token_index >= rows || out_n != 1280) {
        return -1;
    }
    const uint16_t *norm_w = mu_tensor_bf16(e, "visual.blocks.0.norm1.weight", 1, 1280, 0);
    const uint16_t *norm_b = mu_tensor_bf16(e, "visual.blocks.0.norm1.bias", 1, 1280, 0);
    const uint16_t *qkv_w = mu_tensor_bf16(e, "visual.blocks.0.attn.qkv.weight", 2, 3840, 1280);
    const uint16_t *qkv_b = mu_tensor_bf16(e, "visual.blocks.0.attn.qkv.bias", 1, 3840, 0);
    const uint16_t *proj_w = mu_tensor_bf16(e, "visual.blocks.0.attn.proj.weight", 2, 1280, 1280);
    const uint16_t *proj_b = mu_tensor_bf16(e, "visual.blocks.0.attn.proj.bias", 1, 1280, 0);
    if (!norm_w || !norm_b || !qkv_w || !qkv_b || !proj_w || !proj_b) return -2;

    float *normed = (float *)malloc((size_t)rows * 1280u * sizeof(normed[0]));
    float *kv = (float *)malloc((size_t)rows * 2560u * sizeof(kv[0]));
    float *w_tmp = (float *)malloc(2560u * 1280u * sizeof(w_tmp[0]));
    float *scores = (float *)malloc((size_t)rows * sizeof(scores[0]));
    if (!normed || !kv || !w_tmp || !scores) {
        free(normed);
        free(kv);
        free(w_tmp);
        free(scores);
        return -3;
    }

    for (int r = 0; r < rows; r++) {
        if (mu_layernorm_one_bf16(patch_embeds + (size_t)r * 1280u, 1280,
                                  norm_w, norm_b, 1e-6f,
                                  normed + (size_t)r * 1280u) != 0) {
            free(normed);
            free(kv);
            free(w_tmp);
            free(scores);
            return -4;
        }
    }

    float q0[1280];
    if (mu_linear_one_bf16(normed + (size_t)token_index * 1280u,
                           1280, qkv_w, qkv_b, 1280, q0) != 0) {
        free(normed);
        free(kv);
        free(w_tmp);
        free(scores);
        return -5;
    }
    if (mu_linear_seq_f32(normed, rows, 1280,
                          qkv_w + (size_t)1280 * 1280,
                          qkv_b + 1280,
                          2560, kv, w_tmp) != 0) {
        free(normed);
        free(kv);
        free(w_tmp);
        free(scores);
        return -6;
    }
    for (size_t i = 0; i < (size_t)rows * 2560u; i++) {
        kv[i] = mu_bf16_to_f32(mu_f32_to_bf16(kv[i]));
    }

    float attn_concat[1280];
    const float scale = 1.0f / sqrtf(80.0f);
    for (int h = 0; h < 16; h++) {
        float qh[80];
        mu_vision_rope_head(q0 + h * 80, rotary + (size_t)token_index * 40u, qh);
        float max_score = -FLT_MAX;
        for (int r = 0; r < rows; r++) {
            float kh[80];
            mu_vision_rope_head(kv + (size_t)r * 2560u + h * 80,
                                rotary + (size_t)r * 40u, kh);
            float dot = 0.0f;
            for (int d = 0; d < 80; d++) dot += qh[d] * kh[d];
            scores[r] = dot * scale;
            if (scores[r] > max_score) max_score = scores[r];
        }
        float denom = 0.0f;
        for (int r = 0; r < rows; r++) {
            scores[r] = expf(scores[r] - max_score);
            denom += scores[r];
        }
        float *oh = attn_concat + h * 80;
        for (int d = 0; d < 80; d++) oh[d] = 0.0f;
        for (int r = 0; r < rows; r++) {
            float p = mu_bf16_to_f32(mu_f32_to_bf16(scores[r] / denom));
            const float *vh = kv + (size_t)r * 2560u + 1280u + h * 80;
            for (int d = 0; d < 80; d++) oh[d] += p * vh[d];
        }
        for (int d = 0; d < 80; d++) oh[d] = mu_bf16_to_f32(mu_f32_to_bf16(oh[d]));
    }

    int rc = mu_linear_one_bf16(attn_concat, 1280, proj_w, proj_b, 1280, out);
    free(normed);
    free(kv);
    free(w_tmp);
    free(scores);
    return rc == 0 ? 0 : -7;
}

static float mu_quick_gelu(float x) {
    return x / (1.0f + expf(-1.702f * x));
}

int mu_vision_block0_output_token0(mu_engine *e, const float *patch_embeds,
                                   int rows, int cols,
                                   const float *rotary, int rotary_rows, int rotary_cols,
                                   float *out, int out_n) {
    return mu_vision_block0_output_token(e, patch_embeds, rows, cols,
                                         rotary, rotary_rows, rotary_cols,
                                         0, out, out_n);
}

int mu_vision_block0_output_token(mu_engine *e, const float *patch_embeds,
                                  int rows, int cols,
                                  const float *rotary, int rotary_rows, int rotary_cols,
                                  int token_index, float *out, int out_n) {
    if (!e || !patch_embeds || !rotary || !out ||
        rows <= 0 || cols != 1280 ||
        rotary_rows != rows || rotary_cols != 40 ||
        token_index < 0 || token_index >= rows || out_n != 1280) {
        return -1;
    }
    const uint16_t *norm2_w = mu_tensor_bf16(e, "visual.blocks.0.norm2.weight", 1, 1280, 0);
    const uint16_t *norm2_b = mu_tensor_bf16(e, "visual.blocks.0.norm2.bias", 1, 1280, 0);
    const uint16_t *fc1_w = mu_tensor_bf16(e, "visual.blocks.0.mlp.fc1.weight", 2, 5120, 1280);
    const uint16_t *fc1_b = mu_tensor_bf16(e, "visual.blocks.0.mlp.fc1.bias", 1, 5120, 0);
    const uint16_t *fc2_w = mu_tensor_bf16(e, "visual.blocks.0.mlp.fc2.weight", 2, 1280, 5120);
    const uint16_t *fc2_b = mu_tensor_bf16(e, "visual.blocks.0.mlp.fc2.bias", 1, 1280, 0);
    if (!norm2_w || !norm2_b || !fc1_w || !fc1_b || !fc2_w || !fc2_b) return -2;

    float attn[1280];
    if (mu_vision_block0_attn_token(e, patch_embeds, rows, cols,
                                    rotary, rotary_rows, rotary_cols,
                                    token_index, attn, 1280) != 0) {
        return -3;
    }
    float residual1[1280];
    for (int i = 0; i < 1280; i++) {
        residual1[i] = mu_bf16_to_f32(
            mu_f32_to_bf16(patch_embeds[(size_t)token_index * 1280u + i] + attn[i]));
    }

    float norm2[1280];
    if (mu_layernorm_one_bf16(residual1, 1280, norm2_w, norm2_b, 1e-6f, norm2) != 0) {
        return -4;
    }

    float *fc1 = (float *)malloc(5120u * sizeof(fc1[0]));
    float *fc1_act = (float *)malloc(5120u * sizeof(fc1_act[0]));
    if (!fc1 || !fc1_act) {
        free(fc1);
        free(fc1_act);
        return -5;
    }
    if (mu_linear_one_bf16(norm2, 1280, fc1_w, fc1_b, 5120, fc1) != 0) {
        free(fc1);
        free(fc1_act);
        return -6;
    }
    for (int i = 0; i < 5120; i++) {
        fc1_act[i] = mu_bf16_to_f32(mu_f32_to_bf16(mu_quick_gelu(fc1[i])));
    }

    float mlp[1280];
    if (mu_linear_one_bf16(fc1_act, 5120, fc2_w, fc2_b, 1280, mlp) != 0) {
        free(fc1);
        free(fc1_act);
        return -7;
    }
    for (int i = 0; i < 1280; i++) {
        out[i] = mu_bf16_to_f32(mu_f32_to_bf16(residual1[i] + mlp[i]));
    }
    free(fc1);
    free(fc1_act);
    return 0;
}

static const uint16_t *mu_vision_block_tensor_bf16(const mu_engine *e, int layer,
                                                   const char *suffix,
                                                   int ndim, uint64_t d0, uint64_t d1) {
    char name[128];
    int n = snprintf(name, sizeof(name), "visual.blocks.%d.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) return NULL;
    return mu_tensor_bf16(e, name, ndim, d0, d1);
}

static int mu_vision_block_output_all_layer(mu_engine *e, int layer,
                                            const float *patch_embeds,
                                            int rows, int cols,
                                            const float *rotary,
                                            int rotary_rows, int rotary_cols,
                                            float *out, int out_rows, int out_cols) {
    if (!e || !patch_embeds || !rotary || !out ||
        rows <= 0 || cols != 1280 ||
        rotary_rows != rows || rotary_cols != 40 ||
        out_rows != rows || out_cols != 1280 ||
        layer < 0 || layer >= e->cfg.vision_layers) {
        return -1;
    }
    const uint16_t *norm1_w = mu_vision_block_tensor_bf16(e, layer, "norm1.weight", 1, 1280, 0);
    const uint16_t *norm1_b = mu_vision_block_tensor_bf16(e, layer, "norm1.bias", 1, 1280, 0);
    const uint16_t *qkv_w = mu_vision_block_tensor_bf16(e, layer, "attn.qkv.weight", 2, 3840, 1280);
    const uint16_t *qkv_b = mu_vision_block_tensor_bf16(e, layer, "attn.qkv.bias", 1, 3840, 0);
    const uint16_t *proj_w = mu_vision_block_tensor_bf16(e, layer, "attn.proj.weight", 2, 1280, 1280);
    const uint16_t *proj_b = mu_vision_block_tensor_bf16(e, layer, "attn.proj.bias", 1, 1280, 0);
    const uint16_t *norm2_w = mu_vision_block_tensor_bf16(e, layer, "norm2.weight", 1, 1280, 0);
    const uint16_t *norm2_b = mu_vision_block_tensor_bf16(e, layer, "norm2.bias", 1, 1280, 0);
    const uint16_t *fc1_w = mu_vision_block_tensor_bf16(e, layer, "mlp.fc1.weight", 2, 5120, 1280);
    const uint16_t *fc1_b = mu_vision_block_tensor_bf16(e, layer, "mlp.fc1.bias", 1, 5120, 0);
    const uint16_t *fc2_w = mu_vision_block_tensor_bf16(e, layer, "mlp.fc2.weight", 2, 1280, 5120);
    const uint16_t *fc2_b = mu_vision_block_tensor_bf16(e, layer, "mlp.fc2.bias", 1, 1280, 0);
    if (!norm1_w || !norm1_b || !qkv_w || !qkv_b || !proj_w || !proj_b ||
        !norm2_w || !norm2_b || !fc1_w || !fc1_b || !fc2_w || !fc2_b) {
        return -2;
    }

    float *normed = (float *)malloc((size_t)rows * 1280u * sizeof(normed[0]));
    float *qkv = (float *)malloc((size_t)rows * 3840u * sizeof(qkv[0]));
    float *attn = (float *)malloc((size_t)rows * 1280u * sizeof(attn[0]));
    float *proj = (float *)malloc((size_t)rows * 1280u * sizeof(proj[0]));
    float *residual1 = (float *)malloc((size_t)rows * 1280u * sizeof(residual1[0]));
    float *norm2 = (float *)malloc((size_t)rows * 1280u * sizeof(norm2[0]));
    float *fc1 = (float *)malloc((size_t)rows * 5120u * sizeof(fc1[0]));
    float *w_tmp = (float *)malloc(5120u * 1280u * sizeof(w_tmp[0]));
    if (!normed || !qkv || !attn || !proj || !residual1 || !norm2 ||
        !fc1 || !w_tmp) {
        free(normed);
        free(qkv);
        free(attn);
        free(proj);
        free(residual1);
        free(norm2);
        free(fc1);
        free(w_tmp);
        return -3;
    }

    int rc = 0;
    for (int r = 0; r < rows; r++) {
        rc = mu_layernorm_one_bf16(patch_embeds + (size_t)r * 1280u, 1280,
                                   norm1_w, norm1_b, 1e-6f,
                                   normed + (size_t)r * 1280u);
        if (rc) goto fail;
    }
    rc = mu_linear_seq_f32(normed, rows, 1280, qkv_w, qkv_b, 3840, qkv, w_tmp);
    if (rc) goto fail;
    for (size_t i = 0; i < (size_t)rows * 3840u; i++) {
        qkv[i] = mu_bf16_to_f32(mu_f32_to_bf16(qkv[i]));
    }

    rc = mu_vision_attention_all_f32(qkv, rows, rotary, attn);
    if (rc) goto fail;

    rc = mu_linear_seq_f32(attn, rows, 1280, proj_w, proj_b, 1280, proj, w_tmp);
    if (rc) goto fail;
    for (size_t i = 0; i < (size_t)rows * 1280u; i++) {
        proj[i] = mu_bf16_to_f32(mu_f32_to_bf16(proj[i]));
        residual1[i] = mu_bf16_to_f32(mu_f32_to_bf16(patch_embeds[i] + proj[i]));
    }
    for (int r = 0; r < rows; r++) {
        rc = mu_layernorm_one_bf16(residual1 + (size_t)r * 1280u, 1280,
                                   norm2_w, norm2_b, 1e-6f,
                                   norm2 + (size_t)r * 1280u);
        if (rc) goto fail;
    }
    rc = mu_linear_seq_f32(norm2, rows, 1280, fc1_w, fc1_b, 5120, fc1, w_tmp);
    if (rc) goto fail;
    for (size_t i = 0; i < (size_t)rows * 5120u; i++) {
        float v = mu_bf16_to_f32(mu_f32_to_bf16(fc1[i]));
        fc1[i] = mu_bf16_to_f32(mu_f32_to_bf16(mu_quick_gelu(v)));
    }
    rc = mu_linear_seq_f32(fc1, rows, 5120, fc2_w, fc2_b, 1280, proj, w_tmp);
    if (rc) goto fail;
    for (size_t i = 0; i < (size_t)rows * 1280u; i++) {
        float mlp = mu_bf16_to_f32(mu_f32_to_bf16(proj[i]));
        out[i] = mu_bf16_to_f32(mu_f32_to_bf16(residual1[i] + mlp));
    }

fail:
    free(normed);
    free(qkv);
    free(attn);
    free(proj);
    free(residual1);
    free(norm2);
    free(fc1);
    free(w_tmp);
    return rc ? -4 : 0;
}

int mu_vision_block0_output_all(mu_engine *e, const float *patch_embeds,
                                int rows, int cols,
                                const float *rotary, int rotary_rows, int rotary_cols,
                                float *out, int out_rows, int out_cols) {
    return mu_vision_block_output_all_layer(e, 0, patch_embeds, rows, cols,
                                            rotary, rotary_rows, rotary_cols,
                                            out, out_rows, out_cols);
}

static float mu_gelu_f32(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

static int mu_vision_merger(mu_engine *e, const float *hidden,
                            int rows, int cols,
                            float *out, int out_rows, int out_cols) {
    if (!e || !hidden || !out || rows <= 0 || cols != 1280 ||
        rows % 4 != 0 || out_rows != rows / 4 || out_cols != 896) {
        return -1;
    }
    const uint16_t *ln_w = mu_tensor_bf16(e, "visual.merger.ln_q.weight", 1, 1280, 0);
    const uint16_t *ln_b = mu_tensor_bf16(e, "visual.merger.ln_q.bias", 1, 1280, 0);
    const uint16_t *fc0_w = mu_tensor_bf16(e, "visual.merger.mlp.0.weight", 2, 5120, 5120);
    const uint16_t *fc0_b = mu_tensor_bf16(e, "visual.merger.mlp.0.bias", 1, 5120, 0);
    const uint16_t *fc2_w = mu_tensor_bf16(e, "visual.merger.mlp.2.weight", 2, 896, 5120);
    const uint16_t *fc2_b = mu_tensor_bf16(e, "visual.merger.mlp.2.bias", 1, 896, 0);
    if (!ln_w || !ln_b || !fc0_w || !fc0_b || !fc2_w || !fc2_b) return -2;

    int groups = rows / 4;
    float *normed = (float *)malloc((size_t)rows * 1280u * sizeof(normed[0]));
    float *merged = (float *)malloc((size_t)groups * 5120u * sizeof(merged[0]));
    float *hidden5120 = (float *)malloc((size_t)groups * 5120u * sizeof(hidden5120[0]));
    float *w_tmp = (float *)malloc(5120u * 5120u * sizeof(w_tmp[0]));
    if (!normed || !merged || !hidden5120 || !w_tmp) {
        free(normed);
        free(merged);
        free(hidden5120);
        free(w_tmp);
        return -3;
    }

    int rc = 0;
    for (int r = 0; r < rows; r++) {
        rc = mu_layernorm_one_bf16(hidden + (size_t)r * 1280u, 1280,
                                   ln_w, ln_b, 1e-6f,
                                   normed + (size_t)r * 1280u);
        if (rc) goto fail;
    }
    for (int g = 0; g < groups; g++) {
        memcpy(merged + (size_t)g * 5120u,
               normed + (size_t)g * 4u * 1280u,
               5120u * sizeof(merged[0]));
    }

    rc = mu_linear_seq_f32(merged, groups, 5120, fc0_w, fc0_b, 5120, hidden5120, w_tmp);
    if (rc) goto fail;
    for (size_t i = 0; i < (size_t)groups * 5120u; i++) {
        float v = mu_bf16_to_f32(mu_f32_to_bf16(hidden5120[i]));
        hidden5120[i] = mu_bf16_to_f32(mu_f32_to_bf16(mu_gelu_f32(v)));
    }
    rc = mu_linear_seq_f32(hidden5120, groups, 5120, fc2_w, fc2_b, 896, out, w_tmp);
    if (rc) goto fail;
    for (size_t i = 0; i < (size_t)groups * 896u; i++) {
        out[i] = mu_bf16_to_f32(mu_f32_to_bf16(out[i]));
    }

fail:
    free(normed);
    free(merged);
    free(hidden5120);
    free(w_tmp);
    return rc ? -4 : 0;
}

int mu_vision_encode_hidden(mu_engine *e, const float *patch_embeds,
                            int rows, int cols,
                            const float *rotary, int rotary_rows, int rotary_cols,
                            float *out, int out_rows, int out_cols) {
    if (!e || !patch_embeds || !rotary || !out ||
        rows <= 0 || cols != 1280 ||
        rotary_rows != rows || rotary_cols != 40 ||
        out_rows != rows || out_cols != 1280) {
        return -1;
    }
    float *a = (float *)malloc((size_t)rows * 1280u * sizeof(a[0]));
    float *b = (float *)malloc((size_t)rows * 1280u * sizeof(b[0]));
    if (!a || !b) {
        free(a);
        free(b);
        return -2;
    }
    memcpy(a, patch_embeds, (size_t)rows * 1280u * sizeof(a[0]));
    for (int layer = 0; layer < e->cfg.vision_layers; layer++) {
        int rc = mu_vision_block_output_all_layer(e, layer, a, rows, 1280,
                                                  rotary, rows, 40,
                                                  b, rows, 1280);
        if (rc) {
            free(a);
            free(b);
            return -3;
        }
        float *tmp = a;
        a = b;
        b = tmp;
    }
    memcpy(out, a, (size_t)rows * 1280u * sizeof(out[0]));
    free(a);
    free(b);
    return 0;
}

static int mu_cpu_vision_encode(mu_engine *e, const float *patch_embeds,
                                int rows, int cols,
                                const float *rotary, int rotary_rows, int rotary_cols,
                                float *out, int out_rows, int out_cols) {
    if (!e || !patch_embeds || !rotary || !out ||
        rows <= 0 || cols != 1280 ||
        rotary_rows != rows || rotary_cols != 40 ||
        rows % 4 != 0 || out_rows != rows / 4 || out_cols != 896) {
        return -1;
    }
    float *hidden = (float *)malloc((size_t)rows * 1280u * sizeof(hidden[0]));
    if (!hidden) return -2;
    int rc = mu_vision_encode_hidden(e, patch_embeds, rows, cols,
                                     rotary, rotary_rows, rotary_cols,
                                     hidden, rows, 1280);
    if (rc == 0) rc = mu_vision_merger(e, hidden, rows, 1280, out, out_rows, out_cols);
    free(hidden);
    return rc == 0 ? 0 : -3;
}

int mu_vision_encode(mu_engine *e, const float *patch_embeds,
                     int rows, int cols,
                     const float *rotary, int rotary_rows, int rotary_cols,
                     float *out, int out_rows, int out_cols) {
    if (e && e->opt.backend == MU_BACKEND_METAL && e->metal_available) {
        int rc = mu_record_cpu_fallback(e, "vision_encode");
        if (rc) return rc;
    }
    return mu_cpu_vision_encode(e, patch_embeds, rows, cols, rotary,
                                rotary_rows, rotary_cols, out, out_rows, out_cols);
}

static void mu_rmsnorm_seq_bf16(const float *x, const uint16_t *weight,
                                int seq, int hidden, float eps, float *out) {
    for (int s = 0; s < seq; s++) {
        const float *row = x + (size_t)s * (size_t)hidden;
        float *dst = out + (size_t)s * (size_t)hidden;
        float ss = 0.0f;
        for (int i = 0; i < hidden; i++) ss += row[i] * row[i];
        float inv = 1.0f / sqrtf(ss / (float)hidden + eps);
        for (int i = 0; i < hidden; i++) dst[i] = row[i] * inv * mu_bf16_to_f32(weight[i]);
    }
}

static void mu_rmsnorm_one_bf16(const float *x, const uint16_t *weight,
                                int hidden, float eps, float *out) {
    float ss = 0.0f;
    for (int i = 0; i < hidden; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)hidden + eps);
    for (int i = 0; i < hidden; i++) out[i] = x[i] * inv * mu_bf16_to_f32(weight[i]);
}

static int mu_rope_axis_for_dim(int d) {
    if (d < 8) return 0;
    if (d < 20) return 1;
    if (d < 32) return 2;
    if (d < 40) return 0;
    if (d < 52) return 1;
    return 2;
}

static void mu_apply_one_text_rope_head_pos(float *head, const int pos3[3],
                                            const float inv_freq[32]) {
    float old[64];
    memcpy(old, head, sizeof(old));
    for (int d = 0; d < 64; d++) {
        int row = mu_rope_axis_for_dim(d);
        int pos = pos3[row];
        int inv_idx = d < 32 ? d : d - 32;
        float angle = (float)pos * inv_freq[inv_idx];
        float c = cosf(angle);
        float s = sinf(angle);
        float rot = d < 32 ? -old[d + 32] : old[d - 32];
        head[d] = old[d] * c + rot * s;
    }
}

static void mu_apply_text_rope_one(float *q, float *k, const int pos3[3]) {
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const float theta = 1000000.0f;
    float inv_freq[32];
    for (int i = 0; i < 32; i++) {
        inv_freq[i] = powf(theta, -((float)(2 * i) / (float)head_dim));
    }
    for (int h = 0; h < n_heads; h++) {
        mu_apply_one_text_rope_head_pos(q + h * head_dim, pos3, inv_freq);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        mu_apply_one_text_rope_head_pos(k + h * head_dim, pos3, inv_freq);
    }
}

static void mu_apply_one_text_rope_head(float *head, int token_index, int seq, const int *position_ids,
                                        const float inv_freq[32]) {
    float old[64];
    memcpy(old, head, sizeof(old));
    for (int d = 0; d < 64; d++) {
        int row = mu_rope_axis_for_dim(d);
        int pos = position_ids ? position_ids[row * seq + token_index] : token_index;
        int inv_idx = d < 32 ? d : d - 32;
        float angle = (float)pos * inv_freq[inv_idx];
        float c = cosf(angle);
        float s = sinf(angle);
        float rot = d < 32 ? -old[d + 32] : old[d - 32];
        head[d] = old[d] * c + rot * s;
    }
}

static void mu_apply_text_rope(float *q, float *k, int seq, const int *position_ids) {
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const float theta = 1000000.0f;

    float inv_freq[32];
    for (int i = 0; i < 32; i++) {
        inv_freq[i] = powf(theta, -((float)(2 * i) / (float)head_dim));
    }

    for (int t = 0; t < seq; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *head = q + ((size_t)t * n_heads + (size_t)h) * head_dim;
            mu_apply_one_text_rope_head(head, t, seq, position_ids, inv_freq);
        }
        for (int h = 0; h < n_kv_heads; h++) {
            float *head = k + ((size_t)t * n_kv_heads + (size_t)h) * head_dim;
            mu_apply_one_text_rope_head(head, t, seq, position_ids, inv_freq);
        }
    }
}

static int mu_text_attention_f32(const float *q, const float *k, const float *v,
                                 int seq, float *out) {
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int kv_group = 7;
    const int head_dim = 64;
    const float scale = 1.0f / 8.0f;
    float *scores = (float *)malloc((size_t)seq * sizeof(scores[0]));
    if (!scores) return -1;
    for (int t = 0; t < seq; t++) {
        for (int h = 0; h < n_heads; h++) {
            int kvh = h / kv_group;
            const float *qh = q + ((size_t)t * n_heads + (size_t)h) * head_dim;
            float max_score = -FLT_MAX;
            for (int s = 0; s <= t; s++) {
                const float *kh = k + ((size_t)s * n_kv_heads + (size_t)kvh) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                scores[s] = dot * scale;
                if (scores[s] > max_score) max_score = scores[s];
            }
            float denom = 0.0f;
            for (int s = 0; s <= t; s++) {
                scores[s] = expf(scores[s] - max_score);
                denom += scores[s];
            }
            float *oh = out + ((size_t)t * n_heads + (size_t)h) * head_dim;
            for (int d = 0; d < head_dim; d++) oh[d] = 0.0f;
            for (int s = 0; s <= t; s++) {
                float p = scores[s] / denom;
                const float *vh = v + ((size_t)s * n_kv_heads + (size_t)kvh) * head_dim;
                for (int d = 0; d < head_dim; d++) oh[d] += p * vh[d];
            }
        }
    }
    free(scores);
    return 0;
}

static int mu_topk_insert(mu_token_logit *top, int k, int id, float logit) {
    if (k <= 0) return 0;
    if (id < 0) return 0;
    int pos = -1;
    for (int i = 0; i < k; i++) {
        if (top[i].id < 0 || logit > top[i].logit ||
            (logit == top[i].logit && id > top[i].id)) {
            pos = i;
            break;
        }
    }
    if (pos < 0) return 0;
    for (int i = k - 1; i > pos; i--) top[i] = top[i - 1];
    top[pos].id = id;
    top[pos].logit = logit;
    return 1;
}

static int mu_linear_one_f32_tmp(const float *x, int in,
                                 const uint16_t *w_bf16, const uint16_t *bias_bf16,
                                 int out_n, float *out, float *w_tmp) {
    if (!x || !w_bf16 || !out || !w_tmp || in <= 0 || out_n <= 0) return -1;
    mu_bf16_to_f32_array(w_bf16, (size_t)out_n * (size_t)in, w_tmp);
#ifdef __APPLE__
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                out_n, in, 1.0f, w_tmp, in, x, 1, 0.0f, out, 1);
#else
    for (int r = 0; r < out_n; r++) {
        const float *wrow = w_tmp + (size_t)r * (size_t)in;
        float sum = 0.0f;
        for (int c = 0; c < in; c++) sum += x[c] * wrow[c];
        out[r] = sum;
    }
#endif
    if (bias_bf16) {
        for (int r = 0; r < out_n; r++) out[r] += mu_bf16_to_f32(bias_bf16[r]);
    }
    return 0;
}

static int mu_text_top_logits_from_last_hidden(mu_engine *e, const float *hidden_state,
                                              int top_k, mu_token_logit *out) {
    if (!e || !hidden_state || top_k <= 0 || !out) return -1;
    const int hidden = 896;
    const int vocab = 151936;
    const float eps = 1e-6f;
    for (int i = 0; i < top_k; i++) {
        out[i].id = -1;
        out[i].logit = -FLT_MAX;
    }
    const uint16_t *embed = mu_tensor_bf16(e, "model.embed_tokens.weight", 2, vocab, hidden);
    const uint16_t *final_norm = mu_tensor_bf16(e, "model.norm.weight", 1, hidden, 0);
    if (!embed || !final_norm) return -2;
    float last[896];
    mu_rmsnorm_one_bf16(hidden_state, final_norm, hidden, eps, last);
    for (int id = 0; id < vocab; id++) {
        const uint16_t *wrow = embed + (size_t)id * hidden;
        float sum = 0.0f;
        for (int i = 0; i < hidden; i++) sum += last[i] * mu_bf16_to_f32(wrow[i]);
        sum = mu_bf16_to_f32(mu_f32_to_bf16(sum));
        mu_topk_insert(out, top_k, id, sum);
    }
    return top_k;
}

static int mu_text_top_logits_from_embeddings(mu_engine *e, float *hidden_states, int n_ids,
                                              const int *position_ids,
                                              int top_k, mu_token_logit *out) {
    if (!e || !hidden_states || n_ids <= 0 || top_k <= 0 || !out) return -1;
    const int hidden = 896;
    const int inter = 4864;
    const int vocab = 151936;
    const int q_out = 896;
    const int kv_out = 128;
    const float eps = 1e-6f;
    for (int i = 0; i < top_k; i++) {
        out[i].id = -1;
        out[i].logit = -FLT_MAX;
    }

    const uint16_t *embed = mu_tensor_bf16(e, "model.embed_tokens.weight", 2, vocab, hidden);
    const uint16_t *final_norm = mu_tensor_bf16(e, "model.norm.weight", 1, hidden, 0);
    if (!embed || !final_norm) return -2;

    float *residual = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *normed = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *q = (float *)malloc((size_t)n_ids * q_out * sizeof(float));
    float *k = (float *)malloc((size_t)n_ids * kv_out * sizeof(float));
    float *v = (float *)malloc((size_t)n_ids * kv_out * sizeof(float));
    float *attn = (float *)malloc((size_t)n_ids * q_out * sizeof(float));
    float *proj = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *gate = (float *)malloc((size_t)n_ids * inter * sizeof(float));
    float *up = (float *)malloc((size_t)n_ids * inter * sizeof(float));
    float *mid = (float *)malloc((size_t)n_ids * inter * sizeof(float));
    float *w_tmp = (float *)malloc((size_t)inter * hidden * sizeof(float));
    if (!residual || !normed || !q || !k || !v || !attn ||
        !proj || !gate || !up || !mid || !w_tmp) {
        free(residual); free(normed); free(q); free(k); free(v);
        free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
        return -3;
    }

    char name[256];
    for (int layer = 0; layer < 24; layer++) {
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer);
        const uint16_t *input_norm = mu_tensor_bf16(e, name, 1, hidden, 0);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer);
        const uint16_t *post_norm = mu_tensor_bf16(e, name, 1, hidden, 0);
        if (!input_norm || !post_norm) goto fail;

        memcpy(residual, hidden_states, (size_t)n_ids * hidden * sizeof(float));
        mu_rmsnorm_seq_bf16(hidden_states, input_norm, n_ids, hidden, eps, normed);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer);
        const uint16_t *qw = mu_tensor_bf16(e, name, 2, q_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", layer);
        const uint16_t *qb = mu_tensor_bf16(e, name, 1, q_out, 0);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer);
        const uint16_t *kw = mu_tensor_bf16(e, name, 2, kv_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.bias", layer);
        const uint16_t *kb = mu_tensor_bf16(e, name, 1, kv_out, 0);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer);
        const uint16_t *vw = mu_tensor_bf16(e, name, 2, kv_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.bias", layer);
        const uint16_t *vb = mu_tensor_bf16(e, name, 1, kv_out, 0);
        if (!qw || !qb || !kw || !kb || !vw || !vb) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, qw, qb, q_out, q, w_tmp) != 0) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, kw, kb, kv_out, k, w_tmp) != 0) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, vw, vb, kv_out, v, w_tmp) != 0) goto fail;
        mu_apply_text_rope(q, k, n_ids, position_ids);
        if (mu_text_attention_f32(q, k, v, n_ids, attn) != 0) goto fail;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer);
        const uint16_t *ow = mu_tensor_bf16(e, name, 2, hidden, hidden);
        if (!ow) goto fail;
        if (mu_linear_seq_f32(attn, n_ids, hidden, ow, NULL, hidden, proj, w_tmp) != 0) goto fail;
        for (int i = 0; i < n_ids * hidden; i++) hidden_states[i] = residual[i] + proj[i];

        memcpy(residual, hidden_states, (size_t)n_ids * hidden * sizeof(float));
        mu_rmsnorm_seq_bf16(hidden_states, post_norm, n_ids, hidden, eps, normed);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer);
        const uint16_t *gate_w = mu_tensor_bf16(e, name, 2, inter, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer);
        const uint16_t *up_w = mu_tensor_bf16(e, name, 2, inter, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer);
        const uint16_t *down_w = mu_tensor_bf16(e, name, 2, hidden, inter);
        if (!gate_w || !up_w || !down_w) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, gate_w, NULL, inter, gate, w_tmp) != 0) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, up_w, NULL, inter, up, w_tmp) != 0) goto fail;
        for (int i = 0; i < n_ids * inter; i++) mid[i] = mu_silu_f32(gate[i]) * up[i];
        if (mu_linear_seq_f32(mid, n_ids, inter, down_w, NULL, hidden, proj, w_tmp) != 0) goto fail;
        for (int i = 0; i < n_ids * hidden; i++) hidden_states[i] = residual[i] + proj[i];
    }

    mu_rmsnorm_seq_bf16(hidden_states, final_norm, n_ids, hidden, eps, normed);
    const float *last = normed + (size_t)(n_ids - 1) * hidden;
    for (int id = 0; id < vocab; id++) {
        const uint16_t *wrow = embed + (size_t)id * hidden;
        float sum = 0.0f;
        for (int i = 0; i < hidden; i++) sum += last[i] * mu_bf16_to_f32(wrow[i]);
        sum = mu_bf16_to_f32(mu_f32_to_bf16(sum));
        mu_topk_insert(out, top_k, id, sum);
    }

    free(residual); free(normed); free(q); free(k); free(v);
    free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
    return top_k;

fail:
    free(residual); free(normed); free(q); free(k); free(v);
    free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
    return -20;
}

int mu_text_top_logits(mu_engine *e, const int *input_ids, int n_ids,
                       int top_k, mu_token_logit *out) {
    if (!e || !input_ids || n_ids <= 0 || top_k <= 0 || !out) return -1;
    const int hidden = 896;
    const int inter = 4864;
    const int vocab = 151936;
    const int q_out = 896;
    const int kv_out = 128;
    const float eps = 1e-6f;
    for (int i = 0; i < top_k; i++) {
        out[i].id = -1;
        out[i].logit = -FLT_MAX;
    }

    const uint16_t *embed = mu_tensor_bf16(e, "model.embed_tokens.weight", 2, vocab, hidden);
    const uint16_t *final_norm = mu_tensor_bf16(e, "model.norm.weight", 1, hidden, 0);
    if (!embed || !final_norm) return -2;

    float *hidden_states = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *residual = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *normed = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *q = (float *)malloc((size_t)n_ids * q_out * sizeof(float));
    float *k = (float *)malloc((size_t)n_ids * kv_out * sizeof(float));
    float *v = (float *)malloc((size_t)n_ids * kv_out * sizeof(float));
    float *attn = (float *)malloc((size_t)n_ids * q_out * sizeof(float));
    float *proj = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *gate = (float *)malloc((size_t)n_ids * inter * sizeof(float));
    float *up = (float *)malloc((size_t)n_ids * inter * sizeof(float));
    float *mid = (float *)malloc((size_t)n_ids * inter * sizeof(float));
    float *w_tmp = (float *)malloc((size_t)inter * hidden * sizeof(float));
    if (!hidden_states || !residual || !normed || !q || !k || !v || !attn ||
        !proj || !gate || !up || !mid || !w_tmp) {
        free(hidden_states); free(residual); free(normed); free(q); free(k); free(v);
        free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
        return -3;
    }

    for (int s = 0; s < n_ids; s++) {
        int id = input_ids[s];
        if (id < 0 || id >= vocab) {
            free(hidden_states); free(residual); free(normed); free(q); free(k); free(v);
            free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
            return -4;
        }
        const uint16_t *row = embed + (size_t)id * hidden;
        for (int i = 0; i < hidden; i++) hidden_states[(size_t)s * hidden + i] = mu_bf16_to_f32(row[i]);
    }

    char name[256];
    for (int layer = 0; layer < 24; layer++) {
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer);
        const uint16_t *input_norm = mu_tensor_bf16(e, name, 1, hidden, 0);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer);
        const uint16_t *post_norm = mu_tensor_bf16(e, name, 1, hidden, 0);
        if (!input_norm || !post_norm) goto fail;

        memcpy(residual, hidden_states, (size_t)n_ids * hidden * sizeof(float));
        mu_rmsnorm_seq_bf16(hidden_states, input_norm, n_ids, hidden, eps, normed);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer);
        const uint16_t *qw = mu_tensor_bf16(e, name, 2, q_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", layer);
        const uint16_t *qb = mu_tensor_bf16(e, name, 1, q_out, 0);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer);
        const uint16_t *kw = mu_tensor_bf16(e, name, 2, kv_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.bias", layer);
        const uint16_t *kb = mu_tensor_bf16(e, name, 1, kv_out, 0);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer);
        const uint16_t *vw = mu_tensor_bf16(e, name, 2, kv_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.bias", layer);
        const uint16_t *vb = mu_tensor_bf16(e, name, 1, kv_out, 0);
        if (!qw || !qb || !kw || !kb || !vw || !vb) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, qw, qb, q_out, q, w_tmp) != 0) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, kw, kb, kv_out, k, w_tmp) != 0) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, vw, vb, kv_out, v, w_tmp) != 0) goto fail;
        mu_apply_text_rope(q, k, n_ids, NULL);
        if (mu_text_attention_f32(q, k, v, n_ids, attn) != 0) goto fail;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer);
        const uint16_t *ow = mu_tensor_bf16(e, name, 2, hidden, hidden);
        if (!ow) goto fail;
        if (mu_linear_seq_f32(attn, n_ids, hidden, ow, NULL, hidden, proj, w_tmp) != 0) goto fail;
        for (int i = 0; i < n_ids * hidden; i++) hidden_states[i] = residual[i] + proj[i];

        memcpy(residual, hidden_states, (size_t)n_ids * hidden * sizeof(float));
        mu_rmsnorm_seq_bf16(hidden_states, post_norm, n_ids, hidden, eps, normed);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer);
        const uint16_t *gate_w = mu_tensor_bf16(e, name, 2, inter, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer);
        const uint16_t *up_w = mu_tensor_bf16(e, name, 2, inter, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer);
        const uint16_t *down_w = mu_tensor_bf16(e, name, 2, hidden, inter);
        if (!gate_w || !up_w || !down_w) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, gate_w, NULL, inter, gate, w_tmp) != 0) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, up_w, NULL, inter, up, w_tmp) != 0) goto fail;
        for (int i = 0; i < n_ids * inter; i++) mid[i] = mu_silu_f32(gate[i]) * up[i];
        if (mu_linear_seq_f32(mid, n_ids, inter, down_w, NULL, hidden, proj, w_tmp) != 0) goto fail;
        for (int i = 0; i < n_ids * hidden; i++) hidden_states[i] = residual[i] + proj[i];
    }

    mu_rmsnorm_seq_bf16(hidden_states, final_norm, n_ids, hidden, eps, normed);
    const float *last = normed + (size_t)(n_ids - 1) * hidden;
    for (int id = 0; id < vocab; id++) {
        const uint16_t *wrow = embed + (size_t)id * hidden;
        float sum = 0.0f;
        for (int i = 0; i < hidden; i++) sum += last[i] * mu_bf16_to_f32(wrow[i]);
        sum = mu_bf16_to_f32(mu_f32_to_bf16(sum));
        mu_topk_insert(out, top_k, id, sum);
    }

    free(hidden_states); free(residual); free(normed); free(q); free(k); free(v);
    free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
    return top_k;

fail:
    free(hidden_states); free(residual); free(normed); free(q); free(k); free(v);
    free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
    return -20;
}

int mu_text_generate_greedy(mu_engine *e, const int *input_ids, int n_ids,
                            int max_new_tokens, int *out) {
    if (!e || !input_ids || n_ids <= 0 || max_new_tokens <= 0 || !out) return -1;
    int cap = n_ids + max_new_tokens;
    int *ids = (int *)malloc((size_t)cap * sizeof(ids[0]));
    if (!ids) return -2;
    memcpy(ids, input_ids, (size_t)n_ids * sizeof(ids[0]));
    int cur = n_ids;
    int nout = 0;
    for (int step = 0; step < max_new_tokens; step++) {
        mu_token_logit top[8];
        int rc = mu_text_top_logits(e, ids, cur, 8, top);
        if (rc < 1) {
            free(ids);
            return -3;
        }
        float best = top[0].logit;
        int next = top[0].id;
        for (int i = 1; i < 8; i++) {
            if (top[i].id >= 0 && top[i].logit == best && top[i].id < next) next = top[i].id;
        }
        out[nout++] = next;
        ids[cur++] = next;
        if (next == 151645 || next == 151643) break;
    }
    free(ids);
    return nout;
}

int mu_text_top_logits_with_image_embeds(mu_engine *e, const int *input_ids, int n_ids,
                                         const int *position_ids,
                                         const float *image_embeds, int n_image_embeds,
                                         int top_k, mu_token_logit *out) {
    if (!e || !input_ids || n_ids <= 0 || !position_ids || !image_embeds ||
        n_image_embeds <= 0 || top_k <= 0 || !out) {
        return -1;
    }
    const int hidden = 896;
    const int vocab = 151936;
    const uint16_t *embed = mu_tensor_bf16(e, "model.embed_tokens.weight", 2, vocab, hidden);
    if (!embed) return -2;

    float *hidden_states = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    if (!hidden_states) return -3;
    int image_i = 0;
    for (int s = 0; s < n_ids; s++) {
        int id = input_ids[s];
        float *dst = hidden_states + (size_t)s * hidden;
        if (id == 151655) {
            if (image_i >= n_image_embeds) {
                free(hidden_states);
                return -4;
            }
            memcpy(dst, image_embeds + (size_t)image_i * hidden, (size_t)hidden * sizeof(float));
            image_i++;
        } else {
            if (id < 0 || id >= vocab) {
                free(hidden_states);
                return -5;
            }
            const uint16_t *row = embed + (size_t)id * hidden;
            for (int i = 0; i < hidden; i++) dst[i] = mu_bf16_to_f32(row[i]);
        }
    }
    if (image_i != n_image_embeds) {
        free(hidden_states);
        return -6;
    }
    int rc = mu_text_top_logits_from_embeddings(e, hidden_states, n_ids, position_ids, top_k, out);
    free(hidden_states);
    return rc;
}

static size_t mu_text_cache_offset(int layer, int pos, int cache_cap) {
    return ((size_t)layer * (size_t)cache_cap + (size_t)pos) * 128u;
}

static int mu_text_prefill_cache_from_embeddings(mu_engine *e, float *hidden_states,
                                                 int n_ids, const int *position_ids,
                                                 int cache_cap, float *k_cache,
                                                 float *v_cache,
                                                 int top_k, mu_token_logit *out) {
    if (!e || !hidden_states || n_ids <= 0 || !position_ids ||
        cache_cap < n_ids || !k_cache || !v_cache || !out) {
        return -1;
    }
    const int hidden = 896;
    const int inter = 4864;
    const int q_out = 896;
    const int kv_out = 128;
    const float eps = 1e-6f;

    float *residual = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *normed = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *q = (float *)malloc((size_t)n_ids * q_out * sizeof(float));
    float *k = (float *)malloc((size_t)n_ids * kv_out * sizeof(float));
    float *v = (float *)malloc((size_t)n_ids * kv_out * sizeof(float));
    float *attn = (float *)malloc((size_t)n_ids * q_out * sizeof(float));
    float *proj = (float *)malloc((size_t)n_ids * hidden * sizeof(float));
    float *gate = (float *)malloc((size_t)n_ids * inter * sizeof(float));
    float *up = (float *)malloc((size_t)n_ids * inter * sizeof(float));
    float *mid = (float *)malloc((size_t)n_ids * inter * sizeof(float));
    float *w_tmp = (float *)malloc((size_t)inter * hidden * sizeof(float));
    if (!residual || !normed || !q || !k || !v || !attn ||
        !proj || !gate || !up || !mid || !w_tmp) {
        free(residual); free(normed); free(q); free(k); free(v);
        free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
        return -2;
    }

    char name[256];
    for (int layer = 0; layer < 24; layer++) {
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer);
        const uint16_t *input_norm = mu_tensor_bf16(e, name, 1, hidden, 0);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer);
        const uint16_t *post_norm = mu_tensor_bf16(e, name, 1, hidden, 0);
        if (!input_norm || !post_norm) goto fail;

        memcpy(residual, hidden_states, (size_t)n_ids * hidden * sizeof(float));
        mu_rmsnorm_seq_bf16(hidden_states, input_norm, n_ids, hidden, eps, normed);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer);
        const uint16_t *qw = mu_tensor_bf16(e, name, 2, q_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", layer);
        const uint16_t *qb = mu_tensor_bf16(e, name, 1, q_out, 0);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer);
        const uint16_t *kw = mu_tensor_bf16(e, name, 2, kv_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.bias", layer);
        const uint16_t *kb = mu_tensor_bf16(e, name, 1, kv_out, 0);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer);
        const uint16_t *vw = mu_tensor_bf16(e, name, 2, kv_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.bias", layer);
        const uint16_t *vb = mu_tensor_bf16(e, name, 1, kv_out, 0);
        if (!qw || !qb || !kw || !kb || !vw || !vb) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, qw, qb, q_out, q, w_tmp) != 0) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, kw, kb, kv_out, k, w_tmp) != 0) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, vw, vb, kv_out, v, w_tmp) != 0) goto fail;
        mu_apply_text_rope(q, k, n_ids, position_ids);
        for (int s = 0; s < n_ids; s++) {
            memcpy(k_cache + mu_text_cache_offset(layer, s, cache_cap),
                   k + (size_t)s * kv_out, (size_t)kv_out * sizeof(float));
            memcpy(v_cache + mu_text_cache_offset(layer, s, cache_cap),
                   v + (size_t)s * kv_out, (size_t)kv_out * sizeof(float));
        }
        if (mu_text_attention_f32(q, k, v, n_ids, attn) != 0) goto fail;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer);
        const uint16_t *ow = mu_tensor_bf16(e, name, 2, hidden, hidden);
        if (!ow) goto fail;
        if (mu_linear_seq_f32(attn, n_ids, hidden, ow, NULL, hidden, proj, w_tmp) != 0) goto fail;
        for (int i = 0; i < n_ids * hidden; i++) hidden_states[i] = residual[i] + proj[i];

        memcpy(residual, hidden_states, (size_t)n_ids * hidden * sizeof(float));
        mu_rmsnorm_seq_bf16(hidden_states, post_norm, n_ids, hidden, eps, normed);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer);
        const uint16_t *gate_w = mu_tensor_bf16(e, name, 2, inter, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer);
        const uint16_t *up_w = mu_tensor_bf16(e, name, 2, inter, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer);
        const uint16_t *down_w = mu_tensor_bf16(e, name, 2, hidden, inter);
        if (!gate_w || !up_w || !down_w) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, gate_w, NULL, inter, gate, w_tmp) != 0) goto fail;
        if (mu_linear_seq_f32(normed, n_ids, hidden, up_w, NULL, inter, up, w_tmp) != 0) goto fail;
        for (int i = 0; i < n_ids * inter; i++) mid[i] = mu_silu_f32(gate[i]) * up[i];
        if (mu_linear_seq_f32(mid, n_ids, inter, down_w, NULL, hidden, proj, w_tmp) != 0) goto fail;
        for (int i = 0; i < n_ids * hidden; i++) hidden_states[i] = residual[i] + proj[i];
    }

    int rc = mu_text_top_logits_from_last_hidden(e, hidden_states + (size_t)(n_ids - 1) * hidden,
                                                 top_k, out);
    free(residual); free(normed); free(q); free(k); free(v);
    free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
    return rc;

fail:
    free(residual); free(normed); free(q); free(k); free(v);
    free(attn); free(proj); free(gate); free(up); free(mid); free(w_tmp);
    return -20;
}

static int mu_text_attention_one_cached(const float *q, const float *k_cache,
                                        const float *v_cache, int layer,
                                        int cache_len, int cache_cap,
                                        float *out) {
    const int n_heads = 14;
    const int kv_group = 7;
    const int head_dim = 64;
    const float scale = 1.0f / 8.0f;
    float *scores = (float *)malloc((size_t)cache_len * sizeof(scores[0]));
    if (!scores) return -1;
    for (int h = 0; h < n_heads; h++) {
        int kvh = h / kv_group;
        const float *qh = q + h * head_dim;
        float max_score = -FLT_MAX;
        for (int s = 0; s < cache_len; s++) {
            const float *kh = k_cache + mu_text_cache_offset(layer, s, cache_cap) + kvh * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
            scores[s] = dot * scale;
            if (scores[s] > max_score) max_score = scores[s];
        }
        float denom = 0.0f;
        for (int s = 0; s < cache_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            denom += scores[s];
        }
        float *oh = out + h * head_dim;
        for (int d = 0; d < head_dim; d++) oh[d] = 0.0f;
        for (int s = 0; s < cache_len; s++) {
            float p = scores[s] / denom;
            const float *vh = v_cache + mu_text_cache_offset(layer, s, cache_cap) + kvh * head_dim;
            for (int d = 0; d < head_dim; d++) oh[d] += p * vh[d];
        }
    }
    free(scores);
    return 0;
}

static int mu_text_cached_step(mu_engine *e, int token_id, const int pos3[3],
                               int cache_pos, int cache_cap,
                               float *k_cache, float *v_cache,
                               int top_k, mu_token_logit *out) {
    const int hidden = 896;
    const int inter = 4864;
    const int q_out = 896;
    const int kv_out = 128;
    const float eps = 1e-6f;
    const int vocab = 151936;
    const uint16_t *embed = mu_tensor_bf16(e, "model.embed_tokens.weight", 2, vocab, hidden);
    if (!embed || token_id < 0 || token_id >= vocab) return -1;

    float hidden_state[896];
    float residual[896];
    float normed[896];
    float q[896];
    float k[128];
    float v[128];
    float attn[896];
    float proj[896];
    float *gate = (float *)malloc((size_t)inter * sizeof(gate[0]));
    float *up = (float *)malloc((size_t)inter * sizeof(up[0]));
    float *mid = (float *)malloc((size_t)inter * sizeof(mid[0]));
    float *w_tmp = (float *)malloc((size_t)inter * hidden * sizeof(w_tmp[0]));
    if (!gate || !up || !mid || !w_tmp) {
        free(gate); free(up); free(mid); free(w_tmp);
        return -2;
    }
    const uint16_t *row = embed + (size_t)token_id * hidden;
    for (int i = 0; i < hidden; i++) hidden_state[i] = mu_bf16_to_f32(row[i]);

    char name[256];
    for (int layer = 0; layer < 24; layer++) {
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer);
        const uint16_t *input_norm = mu_tensor_bf16(e, name, 1, hidden, 0);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer);
        const uint16_t *post_norm = mu_tensor_bf16(e, name, 1, hidden, 0);
        if (!input_norm || !post_norm) goto fail;

        memcpy(residual, hidden_state, sizeof(residual));
        mu_rmsnorm_one_bf16(hidden_state, input_norm, hidden, eps, normed);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer);
        const uint16_t *qw = mu_tensor_bf16(e, name, 2, q_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.bias", layer);
        const uint16_t *qb = mu_tensor_bf16(e, name, 1, q_out, 0);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer);
        const uint16_t *kw = mu_tensor_bf16(e, name, 2, kv_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.bias", layer);
        const uint16_t *kb = mu_tensor_bf16(e, name, 1, kv_out, 0);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer);
        const uint16_t *vw = mu_tensor_bf16(e, name, 2, kv_out, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.bias", layer);
        const uint16_t *vb = mu_tensor_bf16(e, name, 1, kv_out, 0);
        if (!qw || !qb || !kw || !kb || !vw || !vb) goto fail;
        if (mu_linear_one_f32_tmp(normed, hidden, qw, qb, q_out, q, w_tmp) != 0) goto fail;
        if (mu_linear_one_f32_tmp(normed, hidden, kw, kb, kv_out, k, w_tmp) != 0) goto fail;
        if (mu_linear_one_f32_tmp(normed, hidden, vw, vb, kv_out, v, w_tmp) != 0) goto fail;
        mu_apply_text_rope_one(q, k, pos3);
        memcpy(k_cache + mu_text_cache_offset(layer, cache_pos, cache_cap), k,
               (size_t)kv_out * sizeof(float));
        memcpy(v_cache + mu_text_cache_offset(layer, cache_pos, cache_cap), v,
               (size_t)kv_out * sizeof(float));
        if (mu_text_attention_one_cached(q, k_cache, v_cache, layer,
                                         cache_pos + 1, cache_cap, attn) != 0) goto fail;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer);
        const uint16_t *ow = mu_tensor_bf16(e, name, 2, hidden, hidden);
        if (!ow) goto fail;
        if (mu_linear_one_f32_tmp(attn, hidden, ow, NULL, hidden, proj, w_tmp) != 0) goto fail;
        for (int i = 0; i < hidden; i++) hidden_state[i] = residual[i] + proj[i];

        memcpy(residual, hidden_state, sizeof(residual));
        mu_rmsnorm_one_bf16(hidden_state, post_norm, hidden, eps, normed);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", layer);
        const uint16_t *gate_w = mu_tensor_bf16(e, name, 2, inter, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", layer);
        const uint16_t *up_w = mu_tensor_bf16(e, name, 2, inter, hidden);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", layer);
        const uint16_t *down_w = mu_tensor_bf16(e, name, 2, hidden, inter);
        if (!gate_w || !up_w || !down_w) goto fail;
        if (mu_linear_one_f32_tmp(normed, hidden, gate_w, NULL, inter, gate, w_tmp) != 0) goto fail;
        if (mu_linear_one_f32_tmp(normed, hidden, up_w, NULL, inter, up, w_tmp) != 0) goto fail;
        for (int i = 0; i < inter; i++) mid[i] = mu_silu_f32(gate[i]) * up[i];
        if (mu_linear_one_f32_tmp(mid, inter, down_w, NULL, hidden, proj, w_tmp) != 0) goto fail;
        for (int i = 0; i < hidden; i++) hidden_state[i] = residual[i] + proj[i];
    }

    int rc = mu_text_top_logits_from_last_hidden(e, hidden_state, top_k, out);
    free(gate); free(up); free(mid); free(w_tmp);
    return rc;

fail:
    free(gate); free(up); free(mid); free(w_tmp);
    return -20;
}

static int mu_cpu_text_generate_greedy_with_image_embeds(mu_engine *e,
                                                         const int *input_ids, int n_ids,
                                                         int grid_t, int grid_h, int grid_w,
                                                         const float *image_embeds,
                                                         int n_image_embeds,
                                                         int max_new_tokens, int *out) {
    if (!e || !input_ids || n_ids <= 0 || grid_t <= 0 || grid_h <= 0 || grid_w <= 0 ||
        !image_embeds || n_image_embeds <= 0 || max_new_tokens <= 0 || !out) {
        return -1;
    }
    const int hidden = 896;
    const int layers = 24;
    const int kv_out = 128;
    const int vocab = 151936;
    int cap = n_ids + max_new_tokens;
    int *ids = (int *)malloc((size_t)cap * sizeof(ids[0]));
    int *pos = (int *)malloc((size_t)cap * 3u * sizeof(pos[0]));
    float *hidden_states = (float *)malloc((size_t)n_ids * hidden * sizeof(hidden_states[0]));
    float *k_cache = (float *)calloc((size_t)layers * (size_t)cap * kv_out, sizeof(k_cache[0]));
    float *v_cache = (float *)calloc((size_t)layers * (size_t)cap * kv_out, sizeof(v_cache[0]));
    if (!ids || !pos || !hidden_states || !k_cache || !v_cache) {
        free(ids);
        free(pos);
        free(hidden_states);
        free(k_cache);
        free(v_cache);
        return -2;
    }
    memcpy(ids, input_ids, (size_t)n_ids * sizeof(ids[0]));

    const uint16_t *embed = mu_tensor_bf16(e, "model.embed_tokens.weight", 2, vocab, hidden);
    if (!embed) {
        free(ids); free(pos); free(hidden_states); free(k_cache); free(v_cache);
        return -3;
    }
    int image_i = 0;
    for (int s = 0; s < n_ids; s++) {
        int id = ids[s];
        float *dst = hidden_states + (size_t)s * hidden;
        if (id == 151655) {
            if (image_i >= n_image_embeds) {
                free(ids); free(pos); free(hidden_states); free(k_cache); free(v_cache);
                return -4;
            }
            memcpy(dst, image_embeds + (size_t)image_i * hidden, (size_t)hidden * sizeof(float));
            image_i++;
        } else {
            if (id < 0 || id >= vocab) {
                free(ids); free(pos); free(hidden_states); free(k_cache); free(v_cache);
                return -5;
            }
            const uint16_t *row = embed + (size_t)id * hidden;
            for (int i = 0; i < hidden; i++) dst[i] = mu_bf16_to_f32(row[i]);
        }
    }
    if (image_i != n_image_embeds) {
        free(ids); free(pos); free(hidden_states); free(k_cache); free(v_cache);
        return -6;
    }
    int pos_n = mu_build_position_ids(e, ids, n_ids, grid_t, grid_h, grid_w, pos, n_ids * 3);
    if (pos_n != n_ids * 3) {
        free(ids); free(pos); free(hidden_states); free(k_cache); free(v_cache);
        return -7;
    }

    mu_token_logit top[8];
    int rc = mu_text_prefill_cache_from_embeddings(e, hidden_states, n_ids, pos,
                                                   cap, k_cache, v_cache, 8, top);
    if (rc < 1) {
        free(ids); free(pos); free(hidden_states); free(k_cache); free(v_cache);
        return -8;
    }

    int cur = n_ids;
    int nout = 0;
    for (int step = 0; step < max_new_tokens; step++) {
        float best = top[0].logit;
        int next = top[0].id;
        for (int i = 1; i < 8; i++) {
            if (top[i].id >= 0 && top[i].logit == best && top[i].id < next) next = top[i].id;
        }
        out[nout++] = next;
        ids[cur++] = next;
        if (next == 151645 || next == 151643) break;
        if (step + 1 >= max_new_tokens) break;

        int next_pos_n = mu_build_position_ids(e, ids, cur, grid_t, grid_h, grid_w, pos, cur * 3);
        if (next_pos_n != cur * 3) {
            free(ids); free(pos); free(hidden_states); free(k_cache); free(v_cache);
            return -9;
        }
        int pos3[3] = {
            pos[0 * cur + (cur - 1)],
            pos[1 * cur + (cur - 1)],
            pos[2 * cur + (cur - 1)],
        };
        rc = mu_text_cached_step(e, next, pos3, cur - 1, cap, k_cache, v_cache, 8, top);
        if (rc < 1) {
            free(ids); free(pos); free(hidden_states); free(k_cache); free(v_cache);
            return -10;
        }
    }
    free(ids);
    free(pos);
    free(hidden_states);
    free(k_cache);
    free(v_cache);
    return nout;
}

int mu_text_generate_greedy_with_image_embeds(mu_engine *e,
                                              const int *input_ids, int n_ids,
                                              int grid_t, int grid_h, int grid_w,
                                              const float *image_embeds,
                                              int n_image_embeds,
                                              int max_new_tokens, int *out) {
    if (e && e->opt.backend == MU_BACKEND_METAL && e->metal_available) {
        int rc = mu_record_cpu_fallback(e, "text_generate");
        if (rc) return rc;
    }
    return mu_cpu_text_generate_greedy_with_image_embeds(
        e, input_ids, n_ids, grid_t, grid_h, grid_w, image_embeds,
        n_image_embeds, max_new_tokens, out);
}

mu_engine_options mu_engine_options_default(void) {
    mu_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_dir = "/Users/will/github/mineru-model/models";
    opt.backend = MU_BACKEND_CPU;
    opt.n_threads = 1;
    opt.max_new_tokens = 512;
    opt.allow_cpu_fallback = true;
    return opt;
}

int mu_engine_open(mu_engine **out, const mu_engine_options *opt) {
    if (!out) return -1;
    *out = NULL;
    mu_engine *e = (mu_engine *)calloc(1, sizeof(*e));
    if (!e) return -2;
    e->st.fd = -1;
    e->opt = opt ? *opt : mu_engine_options_default();
#ifdef __APPLE__
    if (e->opt.backend == MU_BACKEND_METAL) {
        int gpu_rc = mu_gpu_create(&e->gpu);
        if (gpu_rc == 0 && mu_gpu_available(e->gpu)) {
            e->metal_available = true;
        } else if (!e->opt.allow_cpu_fallback) {
            mu_engine_close(e);
            return -20;
        } else {
            e->metal_available = false;
            e->cpu_fallback_count++;
        }
    }
#else
    if (e->opt.backend == MU_BACKEND_METAL) {
        if (!e->opt.allow_cpu_fallback) {
            mu_engine_close(e);
            return -20;
        }
        e->metal_available = false;
        e->cpu_fallback_count++;
    }
#endif
    int rc = mu_load_safetensors(e);
    if (rc) {
        mu_engine_close(e);
        return rc;
    }
    rc = mu_validate_config(e);
    if (rc) {
        mu_engine_close(e);
        return rc;
    }
    rc = mu_bind_required_tensors(e);
    if (rc) {
        mu_engine_close(e);
        return rc;
    }
    *out = e;
    return 0;
}

void mu_engine_close(mu_engine *e) {
    if (!e) return;
#ifdef __APPLE__
    mu_gpu_destroy(e->gpu);
#endif
    mu_hash_free(&e->vocab);
    mu_hash_free(&e->merges);
    for (int i = 0; i < 256; i++) free(e->byte_encoder[i]);
    if (e->tensors) {
        for (int i = 0; i < e->tensor_count; i++) free(e->tensors[i].name);
        free(e->tensors);
    }
    mu_safetensors_close(&e->st);
    free(e);
}

void mu_engine_summary(mu_engine *e, FILE *fp) {
    if (!fp) fp = stdout;
    fprintf(fp, "mu backend=%s model_dir=%s\n",
            e && e->opt.backend == MU_BACKEND_METAL ? "metal" : "cpu",
            e && e->opt.model_dir ? e->opt.model_dir : "(null)");
    fprintf(fp, "mu metal_available=%s cpu_fallback_count=%d allow_cpu_fallback=%s\n",
            e && e->metal_available ? "yes" : "no",
            e ? e->cpu_fallback_count : 0,
            e && e->opt.allow_cpu_fallback ? "yes" : "no");
#ifdef __APPLE__
    fprintf(fp, "mu metal_device=%s\n",
            e && e->gpu ? mu_gpu_device_name(e->gpu) : "none");
#else
    fprintf(fp, "mu metal_device=none\n");
#endif
    if (e && e->tensor_count > 0) {
        fprintf(fp, "safetensors tensors=%d dtype=BF16\n", e->tensor_count);
        fprintf(fp, "text_layers=%d hidden_size=%d vision_layers=%d\n",
                e->cfg.text_layers, e->cfg.hidden_size, e->cfg.vision_layers);
        fprintf(fp, "bound_text_layers=%d bound_vision_layers=%d\n",
                e->bound_text_layers, e->bound_vision_layers);
    }
}

int mu_engine_tensor_count(const mu_engine *e) {
    return e ? e->tensor_count : 0;
}

int mu_engine_bf16_tensor_count(const mu_engine *e) {
    return e ? e->bf16_tensor_count : 0;
}

bool mu_engine_metal_available(const mu_engine *e) {
    return e && e->metal_available;
}

int mu_engine_cpu_fallback_count(const mu_engine *e) {
    return e ? e->cpu_fallback_count : 0;
}

int mu_engine_text_layers(const mu_engine *e) {
    return e ? e->cfg.text_layers : 0;
}

int mu_engine_hidden_size(const mu_engine *e) {
    return e ? e->cfg.hidden_size : 0;
}

int mu_engine_vision_layers(const mu_engine *e) {
    return e ? e->cfg.vision_layers : 0;
}

int mu_engine_bound_text_layers(const mu_engine *e) {
    return e ? e->bound_text_layers : 0;
}

int mu_engine_bound_vision_layers(const mu_engine *e) {
    return e ? e->bound_vision_layers : 0;
}

int mu_engine_tensor_index(const mu_engine *e, const char *name) {
    if (!e || !name) return -1;
    for (int i = 0; i < e->tensor_count; i++) {
        if (e->tensors[i].name && strcmp(e->tensors[i].name, name) == 0) return i;
    }
    return -1;
}

int mu_engine_tensor_ndim(const mu_engine *e, int index) {
    if (!e || index < 0 || index >= e->tensor_count) return 0;
    return e->tensors[index].ndim;
}

uint64_t mu_engine_tensor_dim(const mu_engine *e, int index, int dim) {
    if (!e || index < 0 || index >= e->tensor_count) return 0;
    if (dim < 0 || dim >= e->tensors[index].ndim) return 0;
    return e->tensors[index].shape[dim];
}

uint64_t mu_engine_tensor_size_bytes(const mu_engine *e, int index) {
    if (!e || index < 0 || index >= e->tensor_count) return 0;
    return e->tensors[index].nbytes;
}

static float *mu_read_f32_file_exact(const char *path, int count) {
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

static int mu_sb_append(char **buf, size_t *len, size_t *cap, const char *text, size_t n) {
    if (!buf || !len || !cap || !text) return -1;
    if (*len + n + 1 > *cap) {
        size_t next = *cap ? *cap : 256;
        while (*len + n + 1 > next) next *= 2;
        char *p = (char *)realloc(*buf, next);
        if (!p) return -2;
        *buf = p;
        *cap = next;
    }
    memcpy(*buf + *len, text, n);
    *len += n;
    (*buf)[*len] = 0;
    return 0;
}

static void mu_debug_dump_rgb(const char *prefix, const unsigned char *rgb, int width, int height) {
    const char *dir = getenv("MU_DEBUG_CROPS_DIR");
    if (!dir || !*dir || !prefix || !rgb || width <= 0 || height <= 0) return;
    static int counter = 0;
    char path[1024];
    int id = counter++;
    int n = snprintf(path, sizeof(path), "%s/%s-%03d.ppm", dir, prefix, id);
    if (n < 0 || (size_t)n >= sizeof(path)) return;
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(rgb, 3u, (size_t)width * (size_t)height, fp);
    fclose(fp);
}

enum {
    MU_OTSL_NONE = 0,
    MU_OTSL_NL,
    MU_OTSL_FCEL,
    MU_OTSL_ECEL,
    MU_OTSL_LCEL,
    MU_OTSL_UCEL,
    MU_OTSL_XCEL,
};

typedef struct {
    int *tokens;
    char **texts;
    int n;
    int cap;
} mu_otsl_row;

typedef struct {
    mu_otsl_row *rows;
    int n;
    int cap;
} mu_otsl_table;

static int mu_otsl_match_token(const char *p, int *kind, size_t *len) {
    static const struct {
        const char *text;
        int kind;
    } tokens[] = {
        {"<nl>", MU_OTSL_NL},
        {"<fcel>", MU_OTSL_FCEL},
        {"<ecel>", MU_OTSL_ECEL},
        {"<lcel>", MU_OTSL_LCEL},
        {"<ucel>", MU_OTSL_UCEL},
        {"<xcel>", MU_OTSL_XCEL},
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        size_t n = strlen(tokens[i].text);
        if (strncmp(p, tokens[i].text, n) == 0) {
            if (kind) *kind = tokens[i].kind;
            if (len) *len = n;
            return 1;
        }
    }
    return 0;
}

static const char *mu_otsl_find_next_token(const char *p, int *kind, size_t *len) {
    while (p && *p) {
        if (mu_otsl_match_token(p, kind, len)) return p;
        p++;
    }
    return NULL;
}

static void mu_otsl_table_free(mu_otsl_table *table) {
    if (!table) return;
    for (int r = 0; r < table->n; r++) {
        for (int c = 0; c < table->rows[r].n; c++) free(table->rows[r].texts[c]);
        free(table->rows[r].tokens);
        free(table->rows[r].texts);
    }
    free(table->rows);
    memset(table, 0, sizeof(*table));
}

static int mu_otsl_table_add_row(mu_otsl_table *table) {
    if (!table) return -1;
    if (table->n + 1 > table->cap) {
        int next = table->cap ? table->cap * 2 : 8;
        mu_otsl_row *rows = (mu_otsl_row *)realloc(table->rows, (size_t)next * sizeof(rows[0]));
        if (!rows) return -2;
        memset(rows + table->cap, 0, (size_t)(next - table->cap) * sizeof(rows[0]));
        table->rows = rows;
        table->cap = next;
    }
    memset(&table->rows[table->n], 0, sizeof(table->rows[table->n]));
    table->n++;
    return 0;
}

static int mu_otsl_row_append(mu_otsl_row *row, int token, char *text) {
    if (!row) return -1;
    if (row->n + 1 > row->cap) {
        int next = row->cap ? row->cap * 2 : 8;
        int *tokens = (int *)realloc(row->tokens, (size_t)next * sizeof(tokens[0]));
        if (!tokens) return -2;
        row->tokens = tokens;
        char **texts = (char **)realloc(row->texts, (size_t)next * sizeof(texts[0]));
        if (!texts) return -3;
        row->texts = texts;
        for (int i = row->cap; i < next; i++) row->texts[i] = NULL;
        row->cap = next;
    }
    row->tokens[row->n] = token;
    row->texts[row->n] = text;
    row->n++;
    return 0;
}

static char *mu_trimmed_dup(const char *start, const char *end) {
    while (start < end && isspace((unsigned char)*start)) start++;
    while (end > start && isspace((unsigned char)*(end - 1))) end--;
    return mu_strndup_c99(start, (size_t)(end - start));
}

static int mu_otsl_parse(const char *s, mu_otsl_table *table, int *max_cols) {
    if (!s || !table || !max_cols) return -1;
    memset(table, 0, sizeof(*table));
    *max_cols = 0;
    if (mu_otsl_table_add_row(table) != 0) return -2;

    const char *p = s;
    int kind = MU_OTSL_NONE;
    size_t token_len = 0;
    while ((p = mu_otsl_find_next_token(p, &kind, &token_len)) != NULL) {
        p += token_len;
        if (kind == MU_OTSL_NL) {
            if (table->rows[table->n - 1].n > *max_cols) *max_cols = table->rows[table->n - 1].n;
            if (table->rows[table->n - 1].n > 0 && mu_otsl_table_add_row(table) != 0) {
                mu_otsl_table_free(table);
                return -3;
            }
            continue;
        }

        char *text = NULL;
        int next_kind = MU_OTSL_NONE;
        size_t next_len = 0;
        const char *next = mu_otsl_find_next_token(p, &next_kind, &next_len);
        if (kind == MU_OTSL_FCEL) {
            const char *end = next ? next : p + strlen(p);
            text = mu_trimmed_dup(p, end);
            if (!text) {
                mu_otsl_table_free(table);
                return -4;
            }
        }

        if (mu_otsl_row_append(&table->rows[table->n - 1], kind, text) != 0) {
            free(text);
            mu_otsl_table_free(table);
            return -5;
        }
        if (next) p = next;
    }

    if (table->n > 0 && table->rows[table->n - 1].n == 0) table->n--;
    for (int r = 0; r < table->n; r++) {
        if (table->rows[r].n > *max_cols) *max_cols = table->rows[r].n;
    }
    return 0;
}

static int mu_otsl_kind_at(const mu_otsl_table *table, int r, int c) {
    if (!table || r < 0 || r >= table->n || c < 0) return MU_OTSL_ECEL;
    const mu_otsl_row *row = &table->rows[r];
    if (c >= row->n) return MU_OTSL_ECEL;
    return row->tokens[c];
}

static char *mu_otsl_text_at(const mu_otsl_table *table, int r, int c) {
    if (!table || r < 0 || r >= table->n || c < 0) return NULL;
    const mu_otsl_row *row = &table->rows[r];
    if (c >= row->n) return NULL;
    return row->texts[c];
}

static int mu_html_escape_append(char **buf, size_t *len, size_t *cap, const char *text) {
    if (!text) return 0;
    for (const char *p = text; *p; p++) {
        switch (*p) {
        case '&':
            if (mu_sb_append(buf, len, cap, "&amp;", 5) != 0) return -1;
            break;
        case '<':
            if (mu_sb_append(buf, len, cap, "&lt;", 4) != 0) return -2;
            break;
        case '>':
            if (mu_sb_append(buf, len, cap, "&gt;", 4) != 0) return -3;
            break;
        case '"':
            if (mu_sb_append(buf, len, cap, "&quot;", 6) != 0) return -4;
            break;
        case '\'':
            if (mu_sb_append(buf, len, cap, "&#x27;", 6) != 0) return -5;
            break;
        default:
            if (mu_sb_append(buf, len, cap, p, 1) != 0) return -6;
            break;
        }
    }
    return 0;
}

char *mu_otsl_to_html(const char *otsl) {
    if (!otsl) return NULL;
    size_t n = strlen(otsl);
    if (n >= 15 && strncmp(otsl, "<table", 6) == 0 && strcmp(otsl + n - 8, "</table>") == 0) {
        return mu_strdup_c99(otsl);
    }

    mu_otsl_table table;
    int max_cols = 0;
    if (mu_otsl_parse(otsl, &table, &max_cols) != 0) return NULL;
    if (table.n <= 0 || max_cols <= 0) {
        mu_otsl_table_free(&table);
        return mu_strdup_c99("");
    }

    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    int rc = mu_sb_append(&buf, &len, &cap, "<table>", 7);
    for (int r = 0; rc == 0 && r < table.n; r++) {
        rc = mu_sb_append(&buf, &len, &cap, "<tr>", 4);
        for (int c = 0; rc == 0 && c < max_cols; c++) {
            int kind = mu_otsl_kind_at(&table, r, c);
            if (kind != MU_OTSL_FCEL && kind != MU_OTSL_ECEL) continue;

            int colspan = 1;
            while (c + colspan < max_cols) {
                int k = mu_otsl_kind_at(&table, r, c + colspan);
                if (k != MU_OTSL_LCEL && k != MU_OTSL_XCEL) break;
                colspan++;
            }
            int rowspan = 1;
            while (r + rowspan < table.n) {
                int k = mu_otsl_kind_at(&table, r + rowspan, c);
                if (k != MU_OTSL_UCEL && k != MU_OTSL_XCEL) break;
                rowspan++;
            }

            rc = mu_sb_append(&buf, &len, &cap, "<td", 3);
            if (rc == 0 && rowspan > 1) {
                char tmp[32];
                int m = snprintf(tmp, sizeof(tmp), " rowspan=\"%d\"", rowspan);
                if (m < 0 || (size_t)m >= sizeof(tmp)) rc = -10;
                else rc = mu_sb_append(&buf, &len, &cap, tmp, (size_t)m);
            }
            if (rc == 0 && colspan > 1) {
                char tmp[32];
                int m = snprintf(tmp, sizeof(tmp), " colspan=\"%d\"", colspan);
                if (m < 0 || (size_t)m >= sizeof(tmp)) rc = -11;
                else rc = mu_sb_append(&buf, &len, &cap, tmp, (size_t)m);
            }
            if (rc == 0) rc = mu_sb_append(&buf, &len, &cap, ">", 1);
            if (rc == 0 && kind == MU_OTSL_FCEL) {
                rc = mu_html_escape_append(&buf, &len, &cap, mu_otsl_text_at(&table, r, c));
            }
            if (rc == 0) rc = mu_sb_append(&buf, &len, &cap, "</td>", 5);
        }
        if (rc == 0) rc = mu_sb_append(&buf, &len, &cap, "</tr>", 5);
    }
    if (rc == 0) rc = mu_sb_append(&buf, &len, &cap, "</table>", 8);

    mu_otsl_table_free(&table);
    if (rc != 0) {
        free(buf);
        return NULL;
    }
    return buf ? buf : mu_strdup_c99("");
}

static int mu_special_text_for_id(int id, const char **text) {
    for (size_t i = 0; i < sizeof(g_mu_special_tokens) / sizeof(g_mu_special_tokens[0]); i++) {
        if (g_mu_special_tokens[i].id == id) {
            *text = g_mu_special_tokens[i].text;
            return 0;
        }
    }
    return -1;
}

static const char *mu_vocab_key_for_id(const mu_engine *e, int id) {
    if (!e || !e->vocab.slots) return NULL;
    for (size_t i = 0; i < e->vocab.cap; i++) {
        if (e->vocab.slots[i].key && e->vocab.slots[i].value == id) {
            return e->vocab.slots[i].key;
        }
    }
    return NULL;
}

static int mu_append_decoded_vocab_key(mu_engine *e, char **buf, size_t *len,
                                       size_t *cap, const char *key) {
    const char *p = key;
    while (p && *p) {
        int best_b = -1;
        size_t best_n = 0;
        for (int b = 0; b < 256; b++) {
            const char *enc = e->byte_encoder[b];
            if (!enc) continue;
            size_t n = strlen(enc);
            if (n > best_n && strncmp(p, enc, n) == 0) {
                best_b = b;
                best_n = n;
            }
        }
        if (best_b < 0) return -1;
        char c = (char)best_b;
        if (mu_sb_append(buf, len, cap, &c, 1) != 0) return -2;
        p += best_n;
    }
    return 0;
}

static char *mu_decode_token_ids(mu_engine *e, const int *ids, int n_ids) {
    if (!e || !ids || n_ids < 0 || mu_ensure_tokenizer(e) != 0) return NULL;
    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    for (int i = 0; i < n_ids; i++) {
        const char *special = NULL;
        if (mu_special_text_for_id(ids[i], &special) == 0) {
            if (mu_sb_append(&buf, &len, &cap, special, strlen(special)) != 0) {
                free(buf);
                return NULL;
            }
            continue;
        }
        const char *key = mu_vocab_key_for_id(e, ids[i]);
        if (!key || mu_append_decoded_vocab_key(e, &buf, &len, &cap, key) != 0) {
            free(buf);
            return NULL;
        }
    }
    if (!buf) return mu_strdup_c99("");
    return buf;
}

static int mu_json_escape_append(char **buf, size_t *len, size_t *cap, const char *text) {
    if (!text) return mu_sb_append(buf, len, cap, "null", 4);
    if (mu_sb_append(buf, len, cap, "\"", 1) != 0) return -1;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        char tmp[8];
        if (*p == '"' || *p == '\\') {
            tmp[0] = '\\';
            tmp[1] = (char)*p;
            if (mu_sb_append(buf, len, cap, tmp, 2) != 0) return -2;
        } else if (*p == '\n') {
            if (mu_sb_append(buf, len, cap, "\\n", 2) != 0) return -3;
        } else if (*p == '\r') {
            if (mu_sb_append(buf, len, cap, "\\r", 2) != 0) return -4;
        } else if (*p == '\t') {
            if (mu_sb_append(buf, len, cap, "\\t", 2) != 0) return -5;
        } else if (*p < 0x20) {
            snprintf(tmp, sizeof(tmp), "\\u%04x", *p);
            if (mu_sb_append(buf, len, cap, tmp, strlen(tmp)) != 0) return -6;
        } else {
            char c = (char)*p;
            if (mu_sb_append(buf, len, cap, &c, 1) != 0) return -7;
        }
    }
    return mu_sb_append(buf, len, cap, "\"", 1);
}

static char *mu_layout_blocks_to_json(const mu_layout_block *blocks, int n_blocks) {
    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    if (mu_sb_append(&buf, &len, &cap, "[", 1) != 0) return NULL;
    for (int i = 0; i < n_blocks; i++) {
        char tmp[256];
        const mu_layout_block *b = &blocks[i];
        int n = snprintf(tmp, sizeof(tmp),
                         "%s{\"type\":\"%s\",\"bbox\":[%.6g,%.6g,%.6g,%.6g],\"angle\":",
                         i ? "," : "", b->type,
                         b->bbox[0], b->bbox[1], b->bbox[2], b->bbox[3]);
        if (n < 0 || (size_t)n >= sizeof(tmp) ||
            mu_sb_append(&buf, &len, &cap, tmp, (size_t)n) != 0) {
            free(buf);
            return NULL;
        }
        if (b->angle < 0) {
            if (mu_sb_append(&buf, &len, &cap, "null", 4) != 0) {
                free(buf);
                return NULL;
            }
        } else {
            n = snprintf(tmp, sizeof(tmp), "%d", b->angle);
            if (n < 0 || (size_t)n >= sizeof(tmp) ||
                mu_sb_append(&buf, &len, &cap, tmp, (size_t)n) != 0) {
                free(buf);
                return NULL;
            }
        }
        if (mu_sb_append(&buf, &len, &cap, ",\"content\":null", 15) != 0) {
            free(buf);
            return NULL;
        }
        if (b->merge_prev &&
            mu_sb_append(&buf, &len, &cap, ",\"merge_prev\":true", 18) != 0) {
            free(buf);
            return NULL;
        }
        if (mu_sb_append(&buf, &len, &cap, "}", 1) != 0) {
            free(buf);
            return NULL;
        }
    }
    if (mu_sb_append(&buf, &len, &cap, "]", 1) != 0) {
        free(buf);
        return NULL;
    }
    return buf;
}

static char *mu_layout_blocks_to_json_with_content(const mu_layout_block *blocks,
                                                   char **contents,
                                                   int n_blocks) {
    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    if (mu_sb_append(&buf, &len, &cap, "[", 1) != 0) return NULL;
    for (int i = 0; i < n_blocks; i++) {
        char tmp[256];
        const mu_layout_block *b = &blocks[i];
        int n = snprintf(tmp, sizeof(tmp),
                         "%s{\"type\":\"%s\",\"bbox\":[%.6g,%.6g,%.6g,%.6g],\"angle\":",
                         i ? "," : "", b->type,
                         b->bbox[0], b->bbox[1], b->bbox[2], b->bbox[3]);
        if (n < 0 || (size_t)n >= sizeof(tmp) ||
            mu_sb_append(&buf, &len, &cap, tmp, (size_t)n) != 0) {
            free(buf);
            return NULL;
        }
        if (b->angle < 0) {
            if (mu_sb_append(&buf, &len, &cap, "null", 4) != 0) {
                free(buf);
                return NULL;
            }
        } else {
            n = snprintf(tmp, sizeof(tmp), "%d", b->angle);
            if (n < 0 || (size_t)n >= sizeof(tmp) ||
                mu_sb_append(&buf, &len, &cap, tmp, (size_t)n) != 0) {
                free(buf);
                return NULL;
            }
        }
        if (mu_sb_append(&buf, &len, &cap, ",\"content\":", 11) != 0 ||
            mu_json_escape_append(&buf, &len, &cap, contents ? contents[i] : NULL) != 0) {
            free(buf);
            return NULL;
        }
        if (b->merge_prev &&
            mu_sb_append(&buf, &len, &cap, ",\"merge_prev\":true", 18) != 0) {
            free(buf);
            return NULL;
        }
        if (mu_sb_append(&buf, &len, &cap, "}", 1) != 0) {
            free(buf);
            return NULL;
        }
    }
    if (mu_sb_append(&buf, &len, &cap, "]", 1) != 0) {
        free(buf);
        return NULL;
    }
    return buf;
}

static char *mu_layout_contents_to_markdown(char **contents, int n_blocks) {
    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;
    int wrote = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (!contents || !contents[i] || !contents[i][0]) continue;
        if (wrote && mu_sb_append(&buf, &len, &cap, "\n\n", 2) != 0) {
            free(buf);
            return NULL;
        }
        if (mu_sb_append(&buf, &len, &cap, contents[i], strlen(contents[i])) != 0) {
            free(buf);
            return NULL;
        }
        wrote = 1;
    }
    if (!buf) return mu_strdup_c99("");
    return buf;
}

static char *mu_generate_image_region_text(mu_engine *e, const char *path,
                                           const float bbox[4],
                                           const char *task_prompt,
                                           int max_new_tokens) {
    mu_image_tokens tokens;
    memset(&tokens, 0, sizeof(tokens));
    int rc = mu_preprocess_layout_image_region_file(e, path, bbox, &tokens);
    if (rc) return NULL;

    int grid_t = tokens.grid_t;
    int grid_h = tokens.grid_h;
    int grid_w = tokens.grid_w;
    int n_image_embeds = (tokens.grid_t * tokens.grid_h * tokens.grid_w) /
                         (e->cfg.spatial_merge_size * e->cfg.spatial_merge_size);
    float *patch_embeds = (float *)malloc((size_t)tokens.rows * 1280u * sizeof(patch_embeds[0]));
    float *rotary = (float *)malloc((size_t)tokens.rows * 40u * sizeof(rotary[0]));
    float *image_embeds = (float *)malloc((size_t)n_image_embeds * 896u * sizeof(image_embeds[0]));
    if (!patch_embeds || !rotary || !image_embeds) {
        free(patch_embeds);
        free(rotary);
        free(image_embeds);
        mu_image_tokens_free(&tokens);
        return NULL;
    }
    rc = mu_vision_patch_embed(e, &tokens, patch_embeds, tokens.rows, 1280);
    if (rc == 0) rc = mu_vision_rotary_pos_emb(e, tokens.grid_t, tokens.grid_h, tokens.grid_w,
                                               rotary, tokens.rows, 40);
    if (rc == 0) rc = mu_vision_encode(e, patch_embeds, tokens.rows, 1280,
                                       rotary, tokens.rows, 40,
                                       image_embeds, n_image_embeds, 896);
    free(patch_embeds);
    free(rotary);
    mu_image_tokens_free(&tokens);
    if (rc) {
        free(image_embeds);
        return NULL;
    }

    char *prompt = mu_render_chat_prompt(task_prompt, true);
    if (!prompt) {
        free(image_embeds);
        return NULL;
    }
    int cap = 8192;
    int *ids = (int *)malloc((size_t)cap * sizeof(ids[0]));
    int n_ids = ids ? mu_tokenize_image_text(e, prompt, grid_t, grid_h, grid_w, ids, cap) : -1;
    mu_free(prompt);
    if (n_ids <= 0) {
        free(ids);
        free(image_embeds);
        return NULL;
    }
    int *generated = (int *)malloc((size_t)max_new_tokens * sizeof(generated[0]));
    if (!generated) {
        free(ids);
        free(image_embeds);
        return NULL;
    }
    int n_generated = mu_text_generate_greedy_with_image_embeds(
        e, ids, n_ids, grid_t, grid_h, grid_w,
        image_embeds, n_image_embeds, max_new_tokens, generated);
    free(ids);
    free(image_embeds);
    if (n_generated < 0) {
        free(generated);
        return NULL;
    }
    char *text = mu_decode_token_ids(e, generated, n_generated);
    free(generated);
    if (text) {
        char *end = strstr(text, "<|im_end|>");
        if (!end) end = strstr(text, "<|endoftext|>");
        if (end) *end = 0;
    }
    return text;
}

int mu_parse_image_file(mu_engine *e, const char *path, mu_result **out) {
    if (!e || !path || !out) return -1;
    *out = NULL;
    const char *embeds_path = getenv("MU_IMAGE_EMBEDS_FILE");

    mu_image_tokens tokens;
    memset(&tokens, 0, sizeof(tokens));
    int rc = mu_preprocess_layout_image_file(e, path, &tokens);
    if (rc) return -10 + rc;

    const int n_image_embeds = (tokens.grid_t * tokens.grid_h * tokens.grid_w) /
                               (e->cfg.spatial_merge_size * e->cfg.spatial_merge_size);
    float *image_embeds = NULL;
    if (embeds_path && *embeds_path) {
        image_embeds = mu_read_f32_file_exact(embeds_path, n_image_embeds * 896);
    } else {
        float *patch_embeds = (float *)malloc((size_t)tokens.rows * 1280u * sizeof(patch_embeds[0]));
        float *rotary = (float *)malloc((size_t)tokens.rows * 40u * sizeof(rotary[0]));
        image_embeds = (float *)malloc((size_t)n_image_embeds * 896u * sizeof(image_embeds[0]));
        if (!patch_embeds || !rotary || !image_embeds) {
            free(patch_embeds);
            free(rotary);
            free(image_embeds);
            mu_image_tokens_free(&tokens);
            return -20;
        }
        rc = mu_vision_patch_embed(e, &tokens, patch_embeds, tokens.rows, 1280);
        if (rc == 0) {
            rc = mu_vision_rotary_pos_emb(e, tokens.grid_t, tokens.grid_h, tokens.grid_w,
                                          rotary, tokens.rows, 40);
        }
        if (rc == 0) {
            rc = mu_vision_encode(e, patch_embeds, tokens.rows, 1280,
                                  rotary, tokens.rows, 40,
                                  image_embeds, n_image_embeds, 896);
        }
        free(patch_embeds);
        free(rotary);
        if (rc) {
            free(image_embeds);
            mu_image_tokens_free(&tokens);
            return -25 + rc;
        }
    }
    if (!image_embeds) {
        mu_image_tokens_free(&tokens);
        return -20;
    }

    char *prompt = mu_render_chat_prompt("\nLayout Detection:", true);
    if (!prompt) {
        free(image_embeds);
        mu_image_tokens_free(&tokens);
        return -21;
    }
    int cap = 8192;
    int *ids = (int *)malloc((size_t)cap * sizeof(ids[0]));
    if (!ids) {
        mu_free(prompt);
        free(image_embeds);
        mu_image_tokens_free(&tokens);
        return -22;
    }
    int n_ids = mu_tokenize_image_text(e, prompt, tokens.grid_t, tokens.grid_h, tokens.grid_w, ids, cap);
    if (n_ids <= 0) {
        free(ids);
        mu_free(prompt);
        free(image_embeds);
        mu_image_tokens_free(&tokens);
        return -23;
    }
    int max_new = e->opt.max_new_tokens > 0 ? e->opt.max_new_tokens : 512;
    const char *max_new_env = getenv("MU_MAX_NEW_TOKENS");
    if (max_new_env && *max_new_env) {
        long v = strtol(max_new_env, NULL, 10);
        if (v > 0 && v < 4096) max_new = (int)v;
    }
    int *generated = (int *)malloc((size_t)max_new * sizeof(generated[0]));
    if (!generated) {
        free(ids);
        mu_free(prompt);
        free(image_embeds);
        mu_image_tokens_free(&tokens);
        return -24;
    }
    int n_generated = mu_text_generate_greedy_with_image_embeds(
        e, ids, n_ids, tokens.grid_t, tokens.grid_h, tokens.grid_w,
        image_embeds, n_image_embeds, max_new, generated);
    if (n_generated < 0) {
        free(generated);
        free(ids);
        mu_free(prompt);
        free(image_embeds);
        mu_image_tokens_free(&tokens);
        return -30 + n_generated;
    }
    char *raw = mu_decode_token_ids(e, generated, n_generated);
    if (!raw) {
        free(generated);
        free(ids);
        mu_free(prompt);
        free(image_embeds);
        mu_image_tokens_free(&tokens);
        return -40;
    }

    mu_layout_block blocks[256];
    int n_blocks = mu_parse_layout_markup(raw, blocks, 256);
    char **contents = NULL;
    const char *skip_content = getenv("MU_SKIP_CONTENT");
    if (!skip_content || strcmp(skip_content, "1") != 0) {
        contents = (char **)calloc((size_t)n_blocks, sizeof(contents[0]));
        if (!contents && n_blocks > 0) {
            free(raw);
            free(generated);
            free(ids);
            mu_free(prompt);
            free(image_embeds);
            mu_image_tokens_free(&tokens);
            return -41;
        }
        int content_max = max_new;
        const char *content_env = getenv("MU_CONTENT_MAX_NEW_TOKENS");
        if (content_env && *content_env) {
            long v = strtol(content_env, NULL, 10);
            if (v > 0 && v < 4096) content_max = (int)v;
        }
        for (int i = 0; i < n_blocks; i++) {
            const char *task = "\nText Recognition:";
            if (!strcmp(blocks[i].type, "table")) task = "\nTable Recognition:";
            else if (!strcmp(blocks[i].type, "equation") ||
                     !strcmp(blocks[i].type, "equation_block") ||
                     !strcmp(blocks[i].type, "formula_number")) {
                task = "\nFormula Recognition:";
            } else if (!strcmp(blocks[i].type, "image") ||
                       !strcmp(blocks[i].type, "chart")) {
                task = "\nImage Analysis:";
            }
            contents[i] = mu_generate_image_region_text(e, path, blocks[i].bbox, task, content_max);
            if (contents[i] && !strcmp(blocks[i].type, "table")) {
                char *html = mu_otsl_to_html(contents[i]);
                if (html) {
                    free(contents[i]);
                    contents[i] = html;
                }
            }
        }
    }

    char *json = contents ? mu_layout_blocks_to_json_with_content(blocks, contents, n_blocks)
                          : mu_layout_blocks_to_json(blocks, n_blocks);
    char *markdown = contents ? mu_layout_contents_to_markdown(contents, n_blocks)
                              : mu_strdup_c99("");
    mu_result *result = (mu_result *)calloc(1, sizeof(*result));
    if (!json || !markdown || !result) {
        free(result);
        free(json);
        free(markdown);
        if (contents) {
            for (int i = 0; i < n_blocks; i++) free(contents[i]);
            free(contents);
        }
        free(raw);
        free(generated);
        free(ids);
        mu_free(prompt);
        free(image_embeds);
        mu_image_tokens_free(&tokens);
        return -41;
    }
    result->json = json;
    result->markdown = markdown;
    *out = result;

    if (contents) {
        for (int i = 0; i < n_blocks; i++) free(contents[i]);
        free(contents);
    }
    free(raw);
    free(generated);
    free(ids);
    mu_free(prompt);
    free(image_embeds);
    mu_image_tokens_free(&tokens);
    return 0;
}

int mu_parse_image_rgb(mu_engine *e, const uint8_t *rgb, int width, int height,
                       int stride, mu_result **out) {
    (void)e;
    (void)rgb;
    (void)width;
    (void)height;
    (void)stride;
    (void)out;
    return -100;
}

void mu_result_free(mu_result *r) {
    if (!r) return;
    free(r->json);
    free(r->markdown);
    free(r);
}

int mu_result_write_json(const mu_result *r, FILE *fp) {
    if (!r || !fp || !r->json) return -1;
    fputs(r->json, fp);
    return 0;
}

int mu_result_write_markdown(const mu_result *r, FILE *fp) {
    if (!r || !fp || !r->markdown) return -1;
    fputs(r->markdown, fp);
    return 0;
}

char *mu_render_chat_prompt(const char *prompt, bool has_image) {
    static const char *prefix =
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n";
    static const char *image = "<|vision_start|><|image_pad|><|vision_end|>";
    static const char *suffix =
        "<|im_end|>\n"
        "<|im_start|>assistant\n";
    if (!prompt) prompt = "";
    size_t len = strlen(prefix) + strlen(prompt) + strlen(suffix) + 1;
    if (has_image) len += strlen(image);
    char *out = (char *)malloc(len);
    if (!out) return NULL;
    if (has_image) {
        snprintf(out, len, "%s%s%s%s", prefix, image, prompt, suffix);
    } else {
        snprintf(out, len, "%s%s%s", prefix, prompt, suffix);
    }
    return out;
}

void mu_free(void *ptr) {
    free(ptr);
}

static const char *mu_find_token(const char *p, const char *token) {
    return p ? strstr(p, token) : NULL;
}

static int mu_parse_angle(const char *tail, const char *tail_end) {
    size_t len = (size_t)(tail_end - tail);
    if (mu_find_bounded(tail, tail + len, "<|rotate_up|>")) return 0;
    if (mu_find_bounded(tail, tail + len, "<|rotate_right|>")) return 90;
    if (mu_find_bounded(tail, tail + len, "<|rotate_down|>")) return 180;
    if (mu_find_bounded(tail, tail + len, "<|rotate_left|>")) return 270;
    return -1;
}

static bool mu_tail_has_merge_prev(const char *tail, const char *tail_end) {
    size_t len = (size_t)(tail_end - tail);
    return mu_find_bounded(tail, tail + len, "txt_contd_tgt") ||
           mu_find_bounded(tail, tail + len, "<|txt_contd|>");
}

int mu_parse_layout_markup(const char *text, mu_layout_block *blocks, int max_blocks) {
    static const char box_start[] = "<|box_start|>";
    static const char box_end_ref[] = "<|box_end|><|ref_start|>";
    static const char ref_end[] = "<|ref_end|>";
    if (!text || !blocks || max_blocks <= 0) return 0;

    int n = 0;
    const char *p = text;
    while ((p = mu_find_token(p, box_start)) != NULL) {
        p += sizeof(box_start) - 1;
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        int consumed = 0;
        if (sscanf(p, "%d %d %d %d%n", &x1, &y1, &x2, &y2, &consumed) != 4) {
            continue;
        }
        p += consumed;
        const char *mid = mu_find_token(p, box_end_ref);
        if (!mid) break;
        const char *type_start = mid + sizeof(box_end_ref) - 1;
        const char *type_end = mu_find_token(type_start, ref_end);
        if (!type_end) break;
        const char *tail = type_end + sizeof(ref_end) - 1;
        const char *next = mu_find_token(tail, box_start);
        const char *tail_end = next ? next : text + strlen(text);

        if (x1 < 0 || x1 > 1000 || x2 < 0 || x2 > 1000 ||
            y1 < 0 || y1 > 1000 || y2 < 0 || y2 > 1000) {
            p = tail_end;
            continue;
        }
        if (x2 < x1) {
            int tmp = x1;
            x1 = x2;
            x2 = tmp;
        }
        if (y2 < y1) {
            int tmp = y1;
            y1 = y2;
            y2 = tmp;
        }
        if (x1 == x2 || y1 == y2) {
            p = tail_end;
            continue;
        }

        char type[32];
        size_t type_len = (size_t)(type_end - type_start);
        if (type_len >= sizeof(type)) type_len = sizeof(type) - 1;
        memcpy(type, type_start, type_len);
        type[type_len] = 0;
        for (size_t i = 0; i < type_len; i++) {
            if (type[i] >= 'A' && type[i] <= 'Z') type[i] = (char)(type[i] - 'A' + 'a');
        }
        if (!strcmp(type, "inline_formula")) {
            p = tail_end;
            continue;
        }
        if (!strcmp(type, "unknown")) strcpy(type, "image");

        if (n < max_blocks) {
            mu_layout_block *b = &blocks[n++];
            memset(b, 0, sizeof(*b));
            snprintf(b->type, sizeof(b->type), "%s", type);
            b->bbox[0] = (float)x1 / 1000.0f;
            b->bbox[1] = (float)y1 / 1000.0f;
            b->bbox[2] = (float)x2 / 1000.0f;
            b->bbox[3] = (float)y2 / 1000.0f;
            b->angle = mu_parse_angle(tail, tail_end);
            b->merge_prev = !strcmp(type, "text") && mu_tail_has_merge_prev(tail, tail_end);
        }
        p = tail_end;
    }
    return n;
}
