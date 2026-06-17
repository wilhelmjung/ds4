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
int mu_gpu_rmsnorm_probe(mu_gpu *gpu, const float *x, const float *weight,
                         int n, float eps, float *out);
int mu_gpu_vision_encode(mu_gpu *gpu, void *engine,
                         const float *patch_embeds,
                         int rows, int cols,
                         const float *rotary,
                         int rotary_rows, int rotary_cols,
                         float *out, int out_rows, int out_cols);

#endif
