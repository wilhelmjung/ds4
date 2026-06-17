#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


MINERU_ROOT = Path("/Users/will/github/mineru-model")
if str(MINERU_ROOT) not in sys.path:
    sys.path.insert(0, str(MINERU_ROOT))

from test_transformers import limited_sampling_params, load_transformers_model, make_sample_page  # noqa: E402
from mineru_vl_utils import MinerUClient  # noqa: E402
from mineru_vl_utils.mineru_client import DEFAULT_PROMPTS  # noqa: E402


TRACE_VERSION = 1


def tensor_to_list(t: torch.Tensor) -> list[Any]:
    return t.detach().cpu().tolist()


def top_logits(logits: torch.Tensor, k: int) -> list[dict[str, float | int]]:
    values, ids = torch.topk(logits.detach().float().cpu(), k)
    return [
        {"id": int(token_id), "logit": float(value)}
        for token_id, value in zip(ids.tolist(), values.tolist())
    ]


def build_vision_trace(model: Any, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> dict[str, Any]:
    visual = model.model.visual
    hidden_states = visual.patch_embed(pixel_values)
    patch_embeds = hidden_states
    block0_norm1 = visual.blocks[0].norm1(hidden_states)
    block0_qkv_token0 = visual.blocks[0].attn.qkv(block0_norm1[:1])

    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        image_grid_thw[:, 1] * image_grid_thw[:, 2],
        image_grid_thw[:, 0],
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

    block0_attn_output = visual.blocks[0].attn(
        block0_norm1,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
    )
    tiny_rows = min(16, int(hidden_states.shape[0]))
    tiny_rotary = rotary_pos_emb[:tiny_rows]
    tiny_emb = torch.cat((tiny_rotary, tiny_rotary), dim=-1)
    tiny_position_embeddings = (tiny_emb.cos(), tiny_emb.sin())
    tiny_cu_seqlens = torch.tensor(
        [0, tiny_rows],
        dtype=torch.int32,
        device=hidden_states.device,
    )
    block0_tiny_output = visual.blocks[0](
        hidden_states[:tiny_rows].clone(),
        cu_seqlens=tiny_cu_seqlens,
        position_embeddings=tiny_position_embeddings,
    )
    tiny_hidden_states = hidden_states[:tiny_rows].clone()
    for block in visual.blocks:
        tiny_hidden_states = block(
            tiny_hidden_states,
            cu_seqlens=tiny_cu_seqlens,
            position_embeddings=tiny_position_embeddings,
        )
    tiny_image_embeds = visual.merger(tiny_hidden_states)

    block0_output = None
    for idx, block in enumerate(visual.blocks):
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if idx == 0:
            block0_output = hidden_states

    image_embeds = visual.merger(hidden_states)
    return {
        "patch_embeds": patch_embeds,
        "block0_norm1": block0_norm1,
        "block0_qkv_token0": block0_qkv_token0,
        "block0_attn_output": block0_attn_output,
        "rotary_pos_emb": rotary_pos_emb,
        "block0_tiny_output": block0_tiny_output,
        "tiny_last_block_output": tiny_hidden_states,
        "tiny_image_embeds": tiny_image_embeds,
        "block0_output": block0_output,
        "last_block_output": hidden_states,
        "image_embeds": image_embeds,
    }


def build_versions() -> dict[str, str]:
    import transformers
    import mineru_vl_utils

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mineru_vl_utils": getattr(mineru_vl_utils, "__version__", "unknown"),
    }


def build_text_trace(args: argparse.Namespace) -> dict[str, Any]:
    model, processor = load_transformers_model(args.model_dir, args.dtype, args.device)
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        sampling_params=limited_sampling_params(args.max_new_tokens),
        image_analysis=False,
        use_tqdm=False,
    )
    messages = client.client.build_messages(args.prompt, has_image=False)
    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_prompt],
        images=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device=model.device)
    position_ids, rope_deltas = model.model.get_rope_index(
        inputs.input_ids,
        None,
        None,
        inputs.attention_mask,
    )
    with torch.inference_mode():
        outputs = model(
            **inputs,
            position_ids=position_ids,
            use_cache=True,
        )
        generated = model.generate(
            **inputs,
            use_cache=True,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    full_ids = generated[0].detach().cpu().tolist()
    prompt_len = int(inputs.input_ids.shape[1])
    generated_ids = full_ids[prompt_len:]
    generated_text = processor.batch_decode(
        [generated_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    return {
        "trace_version": TRACE_VERSION,
        "mode": "text",
        "model_dir": str(args.model_dir),
        "versions": build_versions(),
        "prompt": args.prompt,
        "chat_prompt": chat_prompt,
        "input_ids": tensor_to_list(inputs.input_ids[0]),
        "attention_mask": tensor_to_list(inputs.attention_mask[0]),
        "position_ids": tensor_to_list(position_ids[:, 0, :]),
        "rope_deltas": tensor_to_list(rope_deltas),
        "top_logits": top_logits(outputs.logits[0, -1], args.top_k),
        "generated_ids": generated_ids,
        "generated_text": generated_text,
    }


def build_layout_trace(args: argparse.Namespace) -> dict[str, Any]:
    model, processor = load_transformers_model(args.model_dir, args.dtype, args.device)
    image_path = args.image
    make_sample_page(image_path)
    image = Image.open(image_path).convert("RGB")
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        sampling_params=limited_sampling_params(args.max_new_tokens),
        image_analysis=args.image_analysis,
        use_tqdm=False,
    )
    layout_image = client.helper.prepare_for_layout(image)
    prompt = DEFAULT_PROMPTS["[layout]"]
    messages = client.client.build_messages(prompt, has_image=True)
    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_prompt],
        images=[layout_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device=model.device, dtype=model.dtype)
    position_ids, rope_deltas = model.model.get_rope_index(
        inputs.input_ids,
        inputs.image_grid_thw,
        None,
        inputs.attention_mask,
    )
    with torch.inference_mode():
        vision_trace = build_vision_trace(model, inputs.pixel_values, inputs.image_grid_thw)
        image_embeds = vision_trace["image_embeds"].to(dtype=model.dtype)
        image_embeds_path = args.out.with_suffix(".image_embeds.f32.bin")
        block0_tiny_path = args.out.with_suffix(".block0_tiny.f32.bin")
        tiny_last_path = args.out.with_suffix(".tiny_last_block.f32.bin")
        tiny_image_embeds_path = args.out.with_suffix(".tiny_image_embeds.f32.bin")
        image_embeds_path.parent.mkdir(parents=True, exist_ok=True)
        image_embeds.detach().float().cpu().numpy().astype("float32").tofile(image_embeds_path)
        vision_trace["block0_tiny_output"].detach().float().cpu().numpy().astype("float32").tofile(
            block0_tiny_path
        )
        vision_trace["tiny_last_block_output"].detach().float().cpu().numpy().astype("float32").tofile(
            tiny_last_path
        )
        vision_trace["tiny_image_embeds"].detach().float().cpu().numpy().astype("float32").tofile(
            tiny_image_embeds_path
        )
        outputs = model(
            **inputs,
            position_ids=position_ids,
            use_cache=True,
        )
        layout_probe_tokens = min(args.max_new_tokens, 4)
        generated = model.generate(
            **inputs,
            use_cache=True,
            do_sample=False,
            max_new_tokens=layout_probe_tokens,
        )
        full_ids = generated[0].detach().cpu().tolist()
        prompt_len = int(inputs.input_ids.shape[1])
        generated_ids = full_ids[prompt_len:]
        generated_text = processor.batch_decode(
            [generated_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        layout_params = client.sampling_params.get("[layout]") or client.sampling_params.get("[default]")
        layout_raw_output = client.client.predict(layout_image, prompt, layout_params)
        layout_result = client.helper.parse_layout_output(layout_raw_output)
    return {
        "trace_version": TRACE_VERSION,
        "mode": "layout",
        "model_dir": str(args.model_dir),
        "versions": build_versions(),
        "image": str(image_path),
        "prompt": prompt,
        "chat_prompt": chat_prompt,
        "input_ids": tensor_to_list(inputs.input_ids[0]),
        "attention_mask": tensor_to_list(inputs.attention_mask[0]),
        "image_grid_thw": tensor_to_list(inputs.image_grid_thw),
        "pixel_values_shape": list(inputs.pixel_values.shape),
        "pixel_values_sample": tensor_to_list(inputs.pixel_values.flatten()[:64]),
        "vision_patch_embeds_shape": list(vision_trace["patch_embeds"].shape),
        "vision_patch_embeds_sample": tensor_to_list(vision_trace["patch_embeds"].flatten()[:64]),
        "vision_rotary_pos_emb_shape": list(vision_trace["rotary_pos_emb"].shape),
        "vision_rotary_pos_emb_sample": tensor_to_list(vision_trace["rotary_pos_emb"].flatten()[:64]),
        "vision_block0_norm1_sample": tensor_to_list(vision_trace["block0_norm1"].flatten()[:64]),
        "vision_block0_qkv_token0_sample": tensor_to_list(vision_trace["block0_qkv_token0"].flatten()[:64]),
        "vision_block0_attn_output_sample": tensor_to_list(vision_trace["block0_attn_output"].flatten()[:64]),
        "vision_block0_output_sample": tensor_to_list(vision_trace["block0_output"].flatten()[:64]),
        "vision_block0_output_token1_sample": tensor_to_list(vision_trace["block0_output"][1].flatten()[:64]),
        "vision_block0_tiny_output_file": str(block0_tiny_path),
        "vision_block0_tiny_output_shape": list(vision_trace["block0_tiny_output"].shape),
        "vision_block0_tiny_output_sample": tensor_to_list(vision_trace["block0_tiny_output"].flatten()[:64]),
        "vision_tiny_last_block_output_file": str(tiny_last_path),
        "vision_tiny_last_block_output_shape": list(vision_trace["tiny_last_block_output"].shape),
        "vision_tiny_last_block_output_sample": tensor_to_list(
            vision_trace["tiny_last_block_output"].flatten()[:64]
        ),
        "vision_tiny_image_embeds_file": str(tiny_image_embeds_path),
        "vision_tiny_image_embeds_shape": list(vision_trace["tiny_image_embeds"].shape),
        "vision_tiny_image_embeds_sample": tensor_to_list(vision_trace["tiny_image_embeds"].flatten()[:64]),
        "vision_last_block_output_sample": tensor_to_list(vision_trace["last_block_output"].flatten()[:64]),
        "image_embeds_file": str(image_embeds_path),
        "image_embeds_shape": list(image_embeds.shape),
        "image_embeds_sample": tensor_to_list(image_embeds.flatten()[:64]),
        "position_ids": tensor_to_list(position_ids[:, 0, :]),
        "rope_deltas": tensor_to_list(rope_deltas),
        "top_logits": top_logits(outputs.logits[0, -1], args.top_k),
        "layout_generated_ids": generated_ids,
        "layout_generated_text": generated_text,
        "layout_raw_output": layout_raw_output,
        "layout_blocks": [dict(block) for block in layout_result],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump MinerU/Qwen2-VL trace artifacts for mu.c.")
    parser.add_argument("--model-dir", type=Path, default=MINERU_ROOT / "models")
    parser.add_argument("--image", type=Path, default=MINERU_ROOT / "sample_page.png")
    parser.add_argument("--mode", choices=["text", "layout"], default="text")
    parser.add_argument("--prompt", default="Respond with OK if the local MinerU model is loaded.")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--image-analysis", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "text":
        trace = build_text_trace(args)
    else:
        trace = build_layout_trace(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
