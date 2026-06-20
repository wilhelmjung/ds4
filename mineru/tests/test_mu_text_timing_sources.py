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


if __name__ == "__main__":
    unittest.main()
