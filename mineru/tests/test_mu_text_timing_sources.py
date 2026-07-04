from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class MuTextTimingSourceTests(unittest.TestCase):
    def test_text_generation_timing_stages_are_wired(self):
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn("typedef struct mu_text_decode_timing", source)
        self.assertIn('"text_generate_prefill"', source)
        self.assertIn('"text_generate_prefill_qkv"', source)
        self.assertIn('"text_generate_prefill_attn"', source)
        self.assertIn('"text_generate_prefill_mlp"', source)
        self.assertIn('"text_generate_prefill_logits"', source)
        self.assertIn('"text_generate_cache_upload"', source)
        self.assertIn('"text_generate_decode"', source)
        self.assertIn('"text_generate_decode_cached_step"', source)
        self.assertIn('"text_generate_decode_cached_qkv"', source)
        self.assertIn('"text_generate_decode_cached_attn_mlp"', source)
        self.assertIn('"text_generate_decode_cached_logits"', source)
        self.assertIn('"text_generate_decode_cached_icb"', source)
        self.assertIn('"text_generate_decode_command_buffers"', source)
        self.assertIn('"text_generate_decode_kernel_dispatches"', source)
        self.assertIn('"text_generate_decode_qkv_dispatches"', source)
        self.assertIn('"text_generate_decode_attn_mlp_dispatches"', source)
        self.assertIn('"text_generate_decode_attention_dispatches"', source)
        self.assertIn('"text_generate_decode_o_proj_dispatches"', source)
        self.assertIn('"text_generate_decode_mlp_dispatches"', source)
        self.assertIn('"text_generate_decode_logits_dispatches"', source)
        self.assertIn("command_buffers", source)
        self.assertIn("kernel_dispatches", source)
        self.assertIn("qkv_dispatches", source)
        self.assertIn("attn_mlp_dispatches", source)
        self.assertIn("attention_dispatches", source)
        self.assertIn("o_proj_dispatches", source)
        self.assertIn("mlp_dispatches", source)
        self.assertIn("logits_dispatches", source)
        self.assertIn("cached_icb", source)
        self.assertIn('getenv("MU_TEXT_DECODE_QKV_ROPE_NO_FUSION")', source)
        self.assertIn('getenv("MU_TEXT_DECODE_ICB_FFN_NO_SIMDGROUP")', source)
        self.assertIn("int qkv_commands = use_qkv_rope_fusion ? 1 : 2;", source)
        self.assertIn("int ffn_dispatches = use_ffn_simdgroup ? 4 : 1;", source)
        self.assertIn("local_timing.kernel_dispatches += (1 + qkv_commands + 1 + 1 + ffn_dispatches) * 24 + 3;", source)
        self.assertIn("typedef struct mu_text_prefill_timing", source)
        self.assertIn("mu_text_prefill_timing_add", source)
        self.assertIn("mu_text_decode_timing_add", source)

    def test_vision_encode_timing_stages_are_wired(self):
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn('"vision_encode_hidden"', source)
        self.assertIn('"vision_encode_merger"', source)

    def test_vision_block_diagnostic_timing_stages_are_wired(self):
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn("MU_VISION_BLOCK_TIMING", source)
        self.assertIn('"vision_block_norm"', source)
        self.assertIn('"vision_block_dense"', source)
        self.assertIn('"vision_block_attention"', source)
        self.assertIn('"vision_block_other"', source)

    def test_text_decode_profile_split_timing_path_is_wired(self):
        source = (ROOT / "mineru/mu.c").read_text()

        self.assertIn('getenv("MU_TEXT_DECODE_PROFILE_SPLIT")', source)
        self.assertIn("mu_gpu_cmd_get_scratch_offsets", source)
        self.assertIn("mu_gpu_cmd_begin_with_scratch_offsets", source)
        self.assertIn("text_decode_profile_qkv", source)
        self.assertIn("text_decode_profile_attention", source)
        self.assertIn("text_decode_profile_o_proj", source)
        self.assertIn("text_decode_profile_mlp", source)
        self.assertIn("text_decode_profile_logits", source)
        self.assertIn('getenv("MU_TEXT_DECODE_ICB") != NULL && !profile_split', source)


if __name__ == "__main__":
    unittest.main()
