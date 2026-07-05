from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class MuMetalKernelSourceTests(unittest.TestCase):
    def test_dense_rows_simd_kernel_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_dense.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_dense_bf16_bias_rows_simd", metal)
        self.assertIn("dense_bf16_bias_rows_simd", host)
        self.assertIn('"mu_dense_bf16_bias_rows_simd"', host)
        self.assertIn('getenv("MU_DENSE_ROWS_SIMD")', host)

    def test_dense_rows_tiled_kernel_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_dense.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_dense_bf16_bias_rows_tiled", metal)
        self.assertIn("dense_bf16_bias_rows_tiled", host)
        self.assertIn('"mu_dense_bf16_bias_rows_tiled"', host)
        self.assertIn('getenv("MU_DENSE_ROWS_TILED")', host)

    def test_dense_rows_2sg_is_default_with_escape_hatch(self):
        metal = (ROOT / "mineru/metal/mu_dense.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_dense_bf16_bias_rows_simdgroup_2sg", metal)
        self.assertIn("kernel void mu_dense_bf16_bias_rows_simdgroup_quick_gelu_2sg", metal)
        self.assertIn("kernel void mu_dense_bf16_bias_rows_simdgroup_qkv_2sg", metal)
        self.assertIn("dense_bf16_bias_rows_simdgroup_2sg", host)
        self.assertIn("dense_bf16_bias_rows_simdgroup_quick_gelu_2sg", host)
        self.assertIn("dense_bf16_bias_rows_simdgroup_qkv_2sg", host)
        self.assertIn('"mu_dense_bf16_bias_rows_simdgroup_2sg"', host)
        self.assertIn('"mu_dense_bf16_bias_rows_simdgroup_quick_gelu_2sg"', host)
        self.assertIn('"mu_dense_bf16_bias_rows_simdgroup_qkv_2sg"', host)
        self.assertIn('getenv("MU_DENSE_ROWS_NO_2SG")', host)
        self.assertIn("(!disable_2sg && gpu->dense_bf16_bias_rows_simdgroup_qkv_2sg)", host)
        self.assertIn("!disable_2sg", host)

    def test_dense_shape_benchmark_compares_mps_cpu_and_current_metal(self):
        bench = (ROOT / "mineru/tests/mu_dense_shape_bench.m").read_text()

        self.assertIn("MPSMatrixMultiplication", bench)
        self.assertIn("cblas_sgemm", bench)
        self.assertIn("mu_gpu_dense_bf16_bias_rows", bench)
        self.assertIn("mps_ms", bench)
        self.assertIn("metal_current_ms", bench)

    def test_dense_rows_mps_bridge_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_dense.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()
        makefile = (ROOT / "Makefile").read_text()

        self.assertIn("mu_dense_mps_bias_round", metal)
        self.assertIn("MPSMatrixMultiplication", host)
        self.assertIn('getenv("MU_DENSE_ROWS_MPS")', host)
        self.assertIn("dense_mps_1280_1280", host)
        self.assertIn("-framework MetalPerformanceShaders", makefile)

    def test_dense_rows_mps_is_default_with_escape_hatch(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn('getenv("MU_DENSE_ROWS_NO_MPS")', host)
        self.assertIn("request_mps || !disable_mps", host)

    def test_dense_f32_rows_mps_diagnostic_path_is_wired(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn('getenv("MU_DENSE_F32_ROWS_MPS")', host)
        self.assertIn('getenv("MU_DENSE_F32_ROWS_NO_MPS")', host)
        self.assertIn("mu_gpu_dense_f32_rows_mps", host)
        self.assertIn("mu_gpu_dense_mps_text_shape", host)
        self.assertIn("request_f32_mps || !disable_f32_mps", host)

    def test_dense_shape_benchmark_can_measure_vision_ops(self):
        bench = (ROOT / "mineru/tests/mu_dense_shape_bench.m").read_text()

        self.assertIn("--vision-ops", bench)
        self.assertIn("mu_gpu_layernorm_bf16_rows", bench)
        self.assertIn("mu_gpu_vision_attn_rows", bench)
        self.assertIn("mu_gpu_vision_quick_gelu_bf16", bench)
        self.assertIn("vision_attn_ms", bench)

    def test_vision_attention_online_kernel_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_vision.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_vision_attn_rows_online", metal)
        self.assertIn("vision_attn_rows_online", host)
        self.assertIn('"mu_vision_attn_rows_online"', host)
        self.assertIn('getenv("MU_VISION_ATTN_ONLINE")', host)

    def test_vision_attention_prerotate_kernel_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_vision.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_vision_rope_qk_rows", metal)
        self.assertIn("kernel void mu_vision_qk_scores_head_prerot", metal)
        self.assertIn("vision_rope_qk_rows", host)
        self.assertIn("vision_qk_scores_head_prerot", host)
        self.assertIn('getenv("MU_VISION_ATTN_PREROTATE")', host)

    def test_vision_attention_prerotate_is_default_with_escape_hatch(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn('getenv("MU_VISION_ATTN_NO_PREROTATE")', host)
        self.assertIn("request_prerotate || !disable_prerotate", host)

    def test_vision_attention_fused_pv_kernel_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_vision.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_vision_softmax_pv_head", metal)
        self.assertIn("vision_softmax_pv_head", host)
        self.assertIn('"mu_vision_softmax_pv_head"', host)
        self.assertIn('getenv("MU_VISION_ATTN_FUSED_PV")', host)
        self.assertIn('getenv("MU_VISION_ATTN_NO_FUSED_PV")', host)
        self.assertIn("request_fused_pv || !disable_fused_pv", host)

    def test_text_cached_rope_update_kernel_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_attn.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()
        header = (ROOT / "mineru/mu_gpu.h").read_text()
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn("kernel void mu_text_rope_cache_update", metal)
        self.assertIn("text_rope_cache_update", host)
        self.assertIn('"mu_text_rope_cache_update"', host)
        self.assertIn("mu_gpu_text_rope_cache_update_ctx", header)
        self.assertIn('getenv("MU_TEXT_CACHED_ROPE_GPU")', source)

    def test_text_cached_layer_resident_path_is_wired(self):
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn('getenv("MU_TEXT_CACHED_LAYER_RESIDENT")', source)
        self.assertIn('getenv("MU_TEXT_CACHED_LAYER_RESIDENT_DISABLE")', source)
        self.assertIn("request_layer_resident || !disable_layer_resident", source)
        self.assertIn("layer_resident", source)
        self.assertIn("cur_hs_buf", source)
        self.assertIn("mu_gpu_text_rope_cache_update_ctx(ctx", source)

    def test_text_cached_hidden_resident_path_is_wired(self):
        header = (ROOT / "mineru/mu_gpu.h").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn("mu_gpu_scratch_b_at", header)
        self.assertIn("mu_gpu_scratch_b_at", host)
        self.assertIn('getenv("MU_TEXT_CACHED_HIDDEN_RESIDENT")', source)
        self.assertIn("hidden_resident", source)
        self.assertIn("hidden_ping", source)

    def test_text_decode_resident_logits_path_is_wired(self):
        header = (ROOT / "mineru/mu_gpu.h").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn("mu_gpu_text_logits_argmax_ctx", header)
        self.assertIn("int mu_gpu_text_logits_argmax_ctx", host)
        self.assertIn('getenv("MU_TEXT_DECODE_NO_RESIDENT_LOGITS")', source)
        self.assertIn("text_decode_resident_logits", source)
        self.assertIn("mu_gpu_text_logits_argmax_ctx(ctx", source)

    def test_text_decoder_projection_fusion_kernels_are_wired(self):
        metal = (ROOT / "mineru/metal/mu_dense.metal").read_text()
        header = (ROOT / "mineru/mu_gpu.h").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn("kernel void mu_text_decode_qkv_proj_simd", metal)
        self.assertIn("kernel void mu_dense_probe_add_simd", metal)
        self.assertIn("mu_gpu_text_decode_qkv_proj_ctx", header)
        self.assertIn("mu_gpu_dense_probe_add_ctx", header)
        self.assertIn('"mu_text_decode_qkv_proj_simd"', host)
        self.assertIn('"mu_dense_probe_add_simd"', host)
        self.assertIn('getenv("MU_TEXT_DECODE_NO_PROBE_ADD_FUSION")', host)
        self.assertIn("mu_gpu_text_decode_qkv_proj_ctx(ctx", source)
        self.assertIn("mu_gpu_dense_probe_add_ctx(ctx", source)

    def test_text_decoder_qkv_rope_cache_fusion_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_attn.metal").read_text()
        header = (ROOT / "mineru/mu_gpu.h").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn("kernel void mu_text_decode_qkv_rope_cache_simd", metal)
        self.assertIn("mu_gpu_text_decode_qkv_rope_cache_ctx", header)
        self.assertIn('"mu_text_decode_qkv_rope_cache_simd"', host)
        self.assertIn('getenv("MU_TEXT_DECODE_QKV_ROPE_FUSION")', source)
        self.assertIn("mu_gpu_text_decode_qkv_rope_cache_ctx(ctx", source)

    def test_icb_qkv_rope_cache_fusion_binds_dynamic_params_in_shader_order(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("[cmd setKernelBuffer:cache->dynamic_params_buf offset:0 atIndex:10]; // pos3", host)
        self.assertIn("[cmd setKernelBuffer:cache->dynamic_params_buf offset:12 atIndex:11]; // cache_pos", host)
        self.assertIn("[cmd setKernelBuffer:gpu->const_hidden_buf offset:0 atIndex:12]; // cols", host)
        self.assertIn("[cmd setKernelBuffer:cache->dynamic_params_buf offset:20 atIndex:13]; // use_bf16_cache", host)
        self.assertIn("[cmd concurrentDispatchThreads:MTLSizeMake(32, 640, 1)", host)

    def test_icb_qkv_rope_cache_fusion_is_default_with_escape_hatch(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn('getenv("MU_TEXT_DECODE_QKV_ROPE_NO_FUSION")', host)
        self.assertIn("!disable_qkv_rope_fusion", host)
        self.assertIn('getenv("MU_TEXT_DECODE_QKV_ROPE_NO_FUSION")', source)
        self.assertIn("int use_qkv_rope_fusion = getenv(\"MU_TEXT_DECODE_QKV_ROPE_NO_FUSION\") == NULL;", source)

    def test_text_decode_ffn_simdgroup_is_default_with_escape_hatch(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn('getenv("MU_TEXT_DECODE_FFN_NO_SIMDGROUP")', host)
        self.assertIn("!disable_simdgroup_ffn", host)
        self.assertIn("mu_gpu_dense_bf16_rows_simdgroup_swiglu_ctx(ctx, normed_buf, gate_w, up_w, 1, 896, 4864, mid_buf)", host)
        self.assertIn("mu_gpu_dense_f32_rows_ctx(ctx, mid_buf, down_w, 1, 4864, 896, proj_buf)", host)

    def test_icb_ffn_simdgroup_sequence_is_default_with_escape_hatch(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn('getenv("MU_TEXT_DECODE_ICB_FFN_NO_SIMDGROUP")', host)
        self.assertIn("!disable_icb_ffn_simdgroup", host)
        self.assertIn("NSUInteger ffn_mid_offset = 0;", host)
        self.assertIn("if (use_ffn_simdgroup) { ffn_mid_offset = offset_a; offset_a += inter_bytes; }", host)
        self.assertIn("[cmd setComputePipelineState:gpu->dense_bf16_rows_simdgroup_swiglu];", host)
        self.assertIn("[cmd setKernelBuffer:cache->dynamic_params_buf offset:24 atIndex:5]; // inter", host)
        self.assertIn("[cmd setKernelBuffer:cache->dynamic_params_buf offset:28 atIndex:6]; // rows", host)
        self.assertIn("[cmd setComputePipelineState:gpu->dense_f32_rows];", host)
        self.assertIn("NSUInteger ffn_commands = use_ffn_simdgroup ? 4 : 1;", host)

    def test_text_cached_attention_simd_is_default_with_escape_hatch(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("text_attn_cached_simd", host)
        self.assertIn('getenv("MU_TEXT_ATTN_CACHED_NO_SIMD")', host)
        self.assertIn("!disable_simd", host)

    def test_text_prefill_flash_attention_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_attn.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_text_prefill_attn_flash", metal)
        self.assertIn("kernel void mu_text_prefill_attn_pos_flash", metal)
        self.assertIn("text_prefill_attn_flash", host)
        self.assertIn("text_prefill_attn_pos_flash", host)
        self.assertIn('getenv("MU_TEXT_PREFILL_ATTN_NO_FLASH")', host)
        self.assertIn("dispatchThreadgroups:grid threadsPerThreadgroup:threads", host)

    def test_latency_profile_uses_metal_gpu_timestamp_selectors(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("@selector(GPUStartTime)", host)
        self.assertIn("@selector(GPUEndTime)", host)

    def test_concurrency_profile_hooks_are_wired(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()
        cli = (ROOT / "mineru/mu_cli.c").read_text()

        self.assertIn('getenv("MU_CONCURRENCY_PROFILE")', host)
        self.assertIn('getenv("MU_CONCURRENCY_PROFILE")', cli)
        self.assertIn("mu_profile stage=concurrency", host)
        self.assertIn("mu_profile stage=concurrency", cli)
        self.assertIn("event=command_wait", host)
        self.assertIn("event=weight_cache", host)
        self.assertIn("event=%s", cli)
        self.assertIn('"parse_start"', cli)
        self.assertIn('"prefetch_join_end"', cli)

    def test_weight_cache_profiles_default_serialized_miss_path(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("mu_gpu_weight_cache_allocate_buffer", host)
        self.assertIn("id<MTLBuffer> buffer = mu_gpu_weight_cache_allocate_buffer", host)
        self.assertIn("mu_gpu_profile_weight_cache(0, stored, length", host)
        self.assertIn("double unlock_time = profile ? local_time_now_seconds() : 0.0;\n    pthread_mutex_unlock(&weight_cache_mutex);", host)
        self.assertNotIn("pthread_cond_wait(&weight_cache_cond", host)
        self.assertNotIn("mu_gpu_weight_cache_pending_index", host)

    def test_command_context_can_preserve_scratch_offsets_for_profile_splits(self):
        header = (ROOT / "mineru/mu_gpu.h").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("mu_gpu_cmd_begin_with_scratch_offsets", header)
        self.assertIn("mu_gpu_cmd_get_scratch_offsets", header)
        self.assertIn("mu_gpu_cmd_begin_with_scratch_offsets", host)
        self.assertIn("mu_gpu_cmd_get_scratch_offsets", host)

    def test_current_vision_encode_has_split_profile_path(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn('getenv("MU_VISION_PROFILE_SPLIT")', host)
        self.assertIn("mu_gpu_profile_commit_stage", host)
        self.assertIn("vision_profile_attention", host)
        self.assertIn("vision_profile_merger_fc2", host)

    def test_vision_attention_shape_profile_is_wired(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn('getenv("MU_VISION_ATTN_SHAPE_PROFILE")', host)
        self.assertIn("vision_attn_shape", host)
        self.assertIn("path=flash", host)
        self.assertIn("threadgroups=", host)

    def test_vision_attention_flash_k16_variant_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_vision.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_vision_attn_rows_flash_k16", metal)
        self.assertIn("vision_attn_rows_flash_k16", host)
        self.assertIn('"mu_vision_attn_rows_flash_k16"', host)
        self.assertIn('getenv("MU_VISION_ATTN_FLASH_K16")', host)
        self.assertIn("key_tile_rows=16", host)

    def test_vision_attention_mpsgraph_lane_is_wired(self):
        makefile = (ROOT / "Makefile").read_text()
        metal = (ROOT / "mineru/metal/mu_vision.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("MetalPerformanceShadersGraph", makefile)
        self.assertIn("#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>", host)
        self.assertIn('getenv("MU_VISION_ATTN_MPSGRAPH")', host)
        self.assertIn('getenv("MU_VISION_ATTN_NO_MPSGRAPH")', host)
        self.assertIn("mu_gpu_vision_attn_rows_mpsgraph_stage", host)
        self.assertIn("scaledDotProductAttentionWithQueryTensor", host)
        self.assertIn("encodeToCommandBuffer", host)
        self.assertIn("path=mpsgraph", host)
        self.assertIn("kernel void mu_vision_attn_pack_qkv_mpsgraph", metal)

    def test_vision_attention_mpsgraph_split_profile_is_wired(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn('getenv("MU_VISION_ATTN_MPSGRAPH_PROFILE")', host)
        self.assertIn("vision_attn_mpsgraph_prepack_boundary", host)
        self.assertIn("vision_attn_mpsgraph_buffer_alloc", host)
        self.assertIn("vision_attn_mpsgraph_pack_qkv", host)
        self.assertIn("vision_attn_mpsgraph_graph", host)
        self.assertIn("vision_attn_mpsgraph_copy_round", host)

    def test_vision_attention_packed_msl_5476_lane_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_vision.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_vision_attn_rows_packed_flash", metal)
        self.assertIn("vision_attn_rows_packed_flash", host)
        self.assertIn('"mu_vision_attn_rows_packed_flash"', host)
        self.assertIn('getenv("MU_VISION_ATTN_MSL_PACKED_5476")', host)
        self.assertIn("rows == 5476", host)
        self.assertIn("path=packed_flash_5476", host)

    def test_vision_layernorm_simd_lane_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_norm.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_layernorm_bf16_rows_simd", metal)
        self.assertIn("layernorm_bf16_rows_simd", host)
        self.assertIn('"mu_layernorm_bf16_rows_simd"', host)
        self.assertIn('getenv("MU_VISION_LAYERNORM_SIMD")', host)
        self.assertIn("cols == 1280", host)

    def test_vision_layernorm_simd_is_default_with_escape_hatch(self):
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn('getenv("MU_VISION_LAYERNORM_NO_SIMD")', host)
        self.assertIn("!disable_simd", host)
