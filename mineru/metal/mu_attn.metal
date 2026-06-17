#include <metal_stdlib>
using namespace metal;

kernel void mu_attn_noop(device float *out [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid == 0) out[0] = out[0];
}
