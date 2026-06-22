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

    def test_text_prefill_flash_attention_is_wired(self):
        metal = (ROOT / "mineru/metal/mu_attn.metal").read_text()
        host = (ROOT / "mineru/mu_metal.m").read_text()

        self.assertIn("kernel void mu_text_prefill_attn_flash", metal)
        self.assertIn("kernel void mu_text_prefill_attn_pos_flash", metal)
        self.assertIn("text_prefill_attn_flash", host)
        self.assertIn("text_prefill_attn_pos_flash", host)
        self.assertIn('getenv("MU_TEXT_PREFILL_ATTN_NO_FLASH")', host)
        self.assertIn("dispatchThreadgroups:grid threadsPerThreadgroup:threads", host)
