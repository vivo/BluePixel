#!/usr/bin/env python3
"""BluePixel 4-step distilled inference with lock-based task competition and resume.

Self-contained single file. No imports from project modules.
Each process (single GPU) competes for sample-level locks on shared storage.
Completed samples are recorded in metadata.jsonl; restart skips them.
No NCCL, no master IP, no node sync beyond shared filesystem.

Usage:
    - Single node, 8 GPUs: launch 8 processes, each with CUDA_VISIBLE_DEVICES=<0..7>
    - Multi-node: launch N processes per node on shared filesystem
    - No node/rank/shard parameters; processes coordinate via file locks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import uuid
from typing import Any

import numpy as np
import torch
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2ImageProcessor, Flux2Pipeline
from diffusers.utils.torch_utils import randn_tensor
from peft import PeftModel
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast

# Try to import model classes from project if available, else from diffusers
try:
    from models.transformer_flux2 import Flux2Transformer2DModel
except ImportError:
    from diffusers import Flux2Transformer2DModel

try:
    from models.autoencoder_dc import AutoencoderDC
except ImportError:
    from diffusers import AutoencoderDC


# ---------------------------------------------------------------------------
# Self-contained helpers (copied from batch_infer_vins.py / infer_fluxklein.py)
# ---------------------------------------------------------------------------

def _get_qwen3_prompt_embeds(
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    prompt: str | list[str],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    max_sequence_length: int = 512,
    hidden_states_layers: list[int] = (9, 18, 27),
):
    dtype = text_encoder.dtype if dtype is None else dtype
    device = text_encoder.device if device is None else device
    prompt = [prompt] if isinstance(prompt, str) else prompt

    texts = []
    for single_prompt in prompt:
        messages = [{"role": "user", "content": single_prompt}]
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ))

    inputs = tokenizer(
        texts, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_sequence_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = text_encoder(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=dtype, device=device)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
    return prompt_embeds


def _prepare_text_ids(x: torch.Tensor, t_coord: torch.Tensor | None = None):
    B, L, _ = x.shape
    out_ids = []
    for i in range(B):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        h = torch.arange(1)
        w = torch.arange(1)
        seq_dim = torch.arange(L)
        coords = torch.cartesian_prod(t, h, w, seq_dim)
        out_ids.append(coords)
    return torch.stack(out_ids)


@torch.no_grad()
def embedding_prompt(
    prompt: str | list[str],
    text_encoder,
    tokenizer,
    device,
    max_sequence_length=512,
    hidden_states_layers=(9, 18, 27),
):
    prompt_embeds = _get_qwen3_prompt_embeds(
        text_encoder=text_encoder, tokenizer=tokenizer, prompt=prompt,
        device=device, max_sequence_length=max_sequence_length,
        hidden_states_layers=hidden_states_layers,
    )
    text_ids = _prepare_text_ids(prompt_embeds).to(device)
    return prompt_embeds, text_ids


def configure_inference_scheduler(scheduler, shift, mu_mode):
    if mu_mode not in ("empirical", "training"):
        raise ValueError(f"Unknown mu_mode: {mu_mode}")
    if shift < 0 or not np.isfinite(shift):
        raise ValueError("shift must be finite and non-negative")
    if mu_mode == "training":
        if shift != 0:
            raise ValueError("mu_mode=training requires shift=0")
        return FlowMatchEulerDiscreteScheduler.from_config(
            scheduler.config, use_dynamic_shifting=True, time_shift_type="exponential"
        )
    if shift > 0:
        return FlowMatchEulerDiscreteScheduler.from_config(
            scheduler.config, use_dynamic_shifting=False, shift=shift
        )
    return FlowMatchEulerDiscreteScheduler.from_config(scheduler.config, use_dynamic_shifting=True)


def compute_inference_mu(image_seq_len, num_steps, mu_mode):
    if mu_mode == "training":
        # Reuse training mu formula: exponential mapping from seq_len
        # Same as utils.model_utils.compute_training_mu
        x = torch.tensor(float(image_seq_len))
        return float((torch.log(1 + x) / (torch.log(torch.tensor(2.0)) + x)).item())
    if mu_mode == "empirical":
        from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
        return float(compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps))
    raise ValueError(f"Unknown mu_mode: {mu_mode}")


def load_base_weights(dit, ckpt_path: str):
    """Load base DiT weights, skipping LoRA keys."""
    from safetensors import safe_open
    model_keys = set(dit.state_dict().keys())
    state_dict = {}
    skipped_lora = skipped_unknown = 0
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "lora_" in key:
                skipped_lora += 1
                continue
            new_key = key.replace("base_model.model.", "").replace(".base_layer", "")
            if new_key in model_keys:
                state_dict[new_key] = f.get_tensor(key)
            else:
                skipped_unknown += 1
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    print(f"[base weights] loaded {len(state_dict)} keys, "
          f"skipped_lora={skipped_lora}, skipped_unknown={skipped_unknown}, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if len(state_dict) == 0:
        raise RuntimeError(f"load_base_weights matched 0 keys from {ckpt_path}")
    if missing:
        print(f"[WARNING] {len(missing)} keys missing (first 10): {list(missing)[:10]}")
    return missing, unexpected


def _load_dcae(src_dir, device, vae_tiling):
    dcae = AutoencoderDC.from_pretrained(src_dir, subfolder="vae").to(device, dtype=torch.float32)
    dcae.requires_grad_(False)
    dcae.eval()
    if vae_tiling:
        dcae.enable_tiling()
        print("  (VAE tiling enabled to save memory)")
    return dcae


def _load_text_pipeline(src_dir, device, dtype):
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        src_dir, subfolder="text_encoder", torch_dtype=dtype,
    ).to(device)
    text_encoder.eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(src_dir, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(src_dir, subfolder="scheduler")
    return text_encoder, tokenizer, scheduler


class FluxKleinPipeline(Flux2Pipeline):
    def __init__(self, transformer, vae, scheduler, text_encoder, tokenizer, vae_scale_factor=64):
        self.register_modules(
            transformer=transformer, scheduler=scheduler, vae=vae,
            text_encoder=text_encoder, tokenizer=tokenizer,
        )
        self.vae_scale_factor = vae_scale_factor
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

    def encode_prompt(self, prompt, device=None, num_images_per_prompt=1, prompt_embeds=None,
                      max_sequence_length=512, text_encoder_out_layers=(9, 18, 27)):
        device = device or self._execution_device
        if prompt is None:
            prompt = ""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embeds, _ = embedding_prompt(
                prompt=prompt, text_encoder=self.text_encoder, tokenizer=self.tokenizer,
                device=device, max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )
        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        text_ids = _prepare_text_ids(prompt_embeds).to(device)
        return prompt_embeds, text_ids

    def _encode_vae_image(self, image, generator):
        latents = self.vae.encode(image)[0]
        latents = latents * self.vae.config.scaling_factor
        return latents

    def prepare_latents(self, batch_size, num_latents_channels, height, width, dtype, device,
                        generator, latents=None):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor
        shape = (batch_size, num_latents_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"generator list length {len(generator)} != batch_size {batch_size}")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) if latents is None \
            else latents.to(device=device, dtype=dtype)
        latent_ids = self._prepare_latent_ids(latents).to(device)
        latents = self._pack_latents(latents)
        return latents, latent_ids

    @torch.no_grad()
    def __call__(self, image=None, instruction=None, prompt=None, prompt_embeds=None,
                 height=None, width=None, num_inference_steps=4, guidance_scale=3.0,
                 num_images_per_prompt=1, generator=None, latents=None, output_type="pil",
                 return_dict=True, attention_kwargs=None, callback_on_step_end=None,
                 callback_on_step_end_tensor_inputs=["latents"], max_sequence_length=512,
                 text_encoder_out_layers=(9, 18, 27), caption_upsample_temperature=None,
                 target_pixels=0, mu_mode="empirical"):
        prompt = instruction or prompt
        self.check_inputs(prompt=prompt, height=height, width=width,
                          prompt_embeds=prompt_embeds,
                          callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs)
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        batch_size = 1 if isinstance(prompt, str) or prompt is None else len(prompt)
        device = self._execution_device

        if prompt_embeds is None:
            prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt, prompt_embeds=prompt_embeds, device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
        else:
            text_ids = self._prepare_text_ids(prompt_embeds)

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt="", device=device, num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        condition_images = None
        if image is not None and not isinstance(image, list):
            image = [image]
        if image is not None:
            condition_images = []
            for img in image:
                img = img.convert("RGB")
                w, h = img.size
                if target_pixels > 0:
                    scale = (target_pixels / (w * h)) ** 0.5
                    w, h = int(w * scale), int(h * scale)
                ow, oh = img.size
                target_ratio = ow / oh
                w_lo, w_hi = max(64, (w - 64) // 64 * 64), (w + 64) // 64 * 64
                h_lo, h_hi = max(64, (h - 64) // 64 * 64), (h + 64) // 64 * 64
                best, best_err = (w, h), 1e9
                for cw in range(w_lo, w_hi + 1, 64):
                    for ch in range(h_lo, h_hi + 1, 64):
                        err = abs(cw / ch - target_ratio)
                        if err < best_err:
                            best_err, best = err, (cw, ch)
                w, h = best
                img = img.resize((w, h), Image.LANCZOS)
                img = self.image_processor.preprocess(img, height=h, width=w)
                condition_images.append(img)
                height = height or h
                width = width or w

        height = height or 1024
        width = width or 1024

        num_channels_latents = self.transformer.config.in_channels
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height, width=width, dtype=prompt_embeds.dtype, device=device,
            generator=generator, latents=latents,
        )

        image_latents = image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator, device=device,
                dtype=self.vae.dtype,
            )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps
        if mu_mode == "training" and (
            not self.scheduler.config.use_dynamic_shifting
            or self.scheduler.config.time_shift_type != "exponential"
        ):
            raise ValueError("Training mu requires an exponential dynamic scheduler")
        mu = compute_inference_mu(image_seq_len, num_inference_steps, mu_mode)
        self.last_mu = mu if self.scheduler.config.use_dynamic_shifting else None
        self.last_mu_mode = mu_mode if self.scheduler.config.use_dynamic_shifting else "fixed"
        self.last_image_seq_len = int(image_seq_len)
        print(f"[Scheduler] mu_mode={self.last_mu_mode} seq_len={image_seq_len} mu={self.last_mu}")
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :latents.size(1)]
                if do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, :latents.size(1)]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents_with_ids(latents, latent_ids)
            latents = latents / self.vae.config.scaling_factor
            latents = latents.to(self.vae.dtype)
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
        return Flux2PipelineOutput(images=image)


def load_pipeline(model_dir, device, vae_tiling, lora_scale, shift, mu_mode, sft_ckpt=None, sft_lora_ckpt=None):
    dtype = torch.bfloat16
    print(f"[Pipeline] Loading from: {model_dir}")
    dcae = _load_dcae(model_dir, device, vae_tiling)
    if sft_ckpt:
        print("[Transformer] Building from pipeline config + SFT weights ...")
        transformer_config = os.path.join(model_dir, "transformer", "config.json")
        dit = Flux2Transformer2DModel.from_config(transformer_config).to(device, dtype)
        dit.requires_grad_(False)
        load_base_weights(dit, sft_ckpt)
        dit.eval()
    else:
        print("[Transformer] Loading from pipeline dir ...")
        dit = Flux2Transformer2DModel.from_pretrained(model_dir, subfolder="transformer").to(device, dtype)
        dit.requires_grad_(False)
        dit.eval()
    if sft_lora_ckpt:
        print(f"[LoRA] Loading adapter: {sft_lora_ckpt}")
        dit = PeftModel.from_pretrained(dit, sft_lora_ckpt)
        if lora_scale is not None and lora_scale >= 0:
            print(f"[LoRA] Setting adapter scale to {lora_scale}")
            for name, module in dit.named_modules():
                if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                    for adapter_name in module.scaling:
                        module.scaling[adapter_name] *= lora_scale
        dit = dit.merge_and_unload()
        dit.eval()
    text_encoder, tokenizer, scheduler = _load_text_pipeline(model_dir, device, dtype)
    scheduler = configure_inference_scheduler(scheduler, shift, mu_mode)
    print(f"[Scheduler] mu_mode={mu_mode}, shift={shift}")
    pipe = FluxKleinPipeline(
        transformer=dit, vae=dcae, scheduler=scheduler,
        text_encoder=text_encoder, tokenizer=tokenizer,
        vae_scale_factor=64,
    )
    pipe.to(device)
    return pipe


def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def make_output_path(out_dir, image_path, instruction, seed=None):
    base = os.path.splitext(os.path.basename(image_path))[0]
    instr_hash = hashlib.md5(instruction.encode("utf-8")).hexdigest()
    fname = f"{base}_{instr_hash}_seed{seed}.jpg" if seed is not None else f"{base}_{instr_hash}.jpg"
    return os.path.join(out_dir, "img", fname)


# ---------------------------------------------------------------------------
# Main inference logic
# ---------------------------------------------------------------------------

def positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value!r}") from None
    if result < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value!r}")
    return result


def nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"must be non-negative: {value!r}") from None
    if result < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative: {value!r}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VINS-Bench inference with lock-based multi-node competition")
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device-id", type=int, default=0,
                   help="CUDA device id. Usually set via CUDA_VISIBLE_DEVICES")
    p.add_argument("--target-pixels", type=positive_int, default=16777216)
    p.add_argument("--steps", type=positive_int, default=4,
                   help="Denoising steps for the distilled model (default: 4)")
    p.add_argument("--guidance", type=nonnegative_float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--shift", type=nonnegative_float, default=0.0)
    p.add_argument("--mu-mode", type=str, choices=["empirical", "training"], default="empirical")
    p.add_argument("--vae-tiling", action="store_true")
    p.add_argument("--lora-scale", type=float, default=None)
    p.add_argument("--sft-ckpt", type=str, default=None)
    p.add_argument("--sft-lora-ckpt", type=str, default=None)
    p.add_argument("--instruction-field", type=str, default="edit_prompt")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--lock-timeout", type=int, default=1800)
    p.add_argument("--metadata-file", type=str, default="metadata.jsonl")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.seeds is not None and not args.seeds.strip():
        p.error("--seeds cannot be empty")
    if args.mu_mode == "training" and args.shift != 0:
        p.error("--mu-mode training requires --shift 0")
    return args


def config_fingerprint(args: argparse.Namespace) -> str:
    cfg = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "jsonl": str(Path(args.jsonl).resolve()),
        "target_pixels": args.target_pixels,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "seeds": args.seeds,
        "shift": args.shift,
        "mu_mode": args.mu_mode,
        "vae_tiling": args.vae_tiling,
        "lora_scale": args.lora_scale,
        "instruction_field": args.instruction_field,
        "sft_ckpt": args.sft_ckpt,
        "sft_lora_ckpt": args.sft_lora_ckpt,
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir).resolve()
    name = Path(args.model_dir).resolve().name.replace("-", "_")
    return Path("evaluation_results") / "vins_bench_v2" / f"{name}_{config_fingerprint(args)[:12]}"


def task_id(sample: dict[str, Any], seed: int, instruction_field: str = "edit_prompt") -> str:
    instruction = sample.get(instruction_field, sample.get("edit_prompt", ""))
    return hashlib.sha256(f"{sample['image_name']}|{instruction}|{seed}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# File-lock based task queue (inspired by test_qwen36_reward_universal_used.py)
# ---------------------------------------------------------------------------
_DIR_CACHE = {}


def _lock_done_dirs(out_dir: Path):
    base = str(out_dir.resolve())
    dirs = _DIR_CACHE.get(base)
    if dirs is None:
        lock_dir = out_dir / "_locks"
        done_dir = out_dir / "_done"
        worker_dir = out_dir / "_workers"
        for d in (lock_dir, done_dir, worker_dir):
            d.mkdir(parents=True, exist_ok=True)
        dirs = (lock_dir, done_dir, worker_dir)
        _DIR_CACHE[base] = dirs
    return dirs


def paths_for_key(out_dir: Path, key: str):
    lock_dir, done_dir, _ = _lock_done_dirs(out_dir)
    return lock_dir / f"{key}.lock", done_dir / f"{key}.done"


def is_done(done_path: Path) -> bool:
    try:
        with done_path.open(encoding="utf-8") as f:
            return json.load(f).get("status") == "success"
    except (OSError, ValueError, AttributeError):
        return False


def load_done_snapshot(out_dir: Path, worker_id: str = "") -> set[str]:
    _, done_dir, _ = _lock_done_dirs(out_dir)
    keys = set()
    t0 = time.time()
    try:
        for entry in os.scandir(done_dir):
            if entry.name.endswith(".done") and is_done(Path(entry.path)):
                keys.add(entry.name[:-len(".done")])
    except OSError as e:
        print(f"[{worker_id}] warning 读取 _done 快照失败，回退逐条检查: {e}")
        return set()
    if keys:
        print(f"[{worker_id}] _done 快照载入 {len(keys)} 条，用时 {time.time() - t0:.1f}s")
    return keys


def try_acquire_lock(lock_path: Path, timeout: int) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\nhost={os.uname().nodename}\ntime={time.time()}\n")
        return True
    except FileExistsError:
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age > timeout:
                lock_path.unlink(missing_ok=True)
                return try_acquire_lock(lock_path, timeout)
        except OSError:
            pass
        return False
    except OSError as e:
        print(f"warning 创建锁文件失败 {lock_path}: {e}")
        return False


def _parse_lock_meta(lock_path: Path):
    meta = {"pid": None, "host": None}
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("pid="):
                    try:
                        meta["pid"] = int(line[len("pid="):])
                    except ValueError:
                        pass
                elif line.startswith("host="):
                    meta["host"] = line[len("host="):]
    except (FileNotFoundError, OSError):
        pass
    return meta


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def cleanup_orphan_locks(out_dir: Path, worker_id: str = "") -> int:
    lock_dir, _, _ = _lock_done_dirs(out_dir)
    this_host = os.uname().nodename
    removed = 0
    try:
        lock_files = list(lock_dir.glob("*.lock"))
    except OSError:
        return 0
    for lp in lock_files:
        meta = _parse_lock_meta(lp)
        if meta["host"] != this_host:
            continue
        if _pid_alive(meta["pid"]):
            continue
        try:
            lp.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            pass
    if removed:
        print(f"[{worker_id}] 启动清理本机孤儿锁 {removed} 个。")
    return removed


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def mark_done(done_path: Path, payload: dict[str, Any]) -> None:
    tmp = done_path.parent / f".{done_path.name}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, done_path)


def load_done_tasks(metadata_path: Path) -> set[str]:
    done = set()
    if not metadata_path.exists():
        return done
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "success":
                    done.add(rec["task_id"])
                elif rec.get("status") == "failed":
                    done.discard(rec["task_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _metadata_lock_path(metadata_path: Path) -> Path:
    return metadata_path.parent / ".metadata.lock"


def append_metadata(metadata_path: Path, record: dict[str, Any]) -> None:
    lock_path = _metadata_lock_path(metadata_path)
    # Simple spin lock on shared storage to prevent interleaved writes.
    while not try_acquire_lock(lock_path, timeout=60):
        time.sleep(0.01)
    try:
        with metadata_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    finally:
        release_lock(lock_path)


def main() -> None:
    args = parse_args()
    out_dir = resolve_out_dir(args)
    worker_id = f"worker_{os.getpid()}"
    device = args.device_id

    if args.check:
        print(json.dumps({
            "torch": torch.__version__,
            "cuda_devices": torch.cuda.device_count(),
            "device": device,
            "worker_id": worker_id,
        }, indent=2))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / args.metadata_file
    _lock_done_dirs(out_dir)
    cleanup_orphan_locks(out_dir, worker_id)

    samples = load_jsonl(args.jsonl)
    if args.limit:
        samples = samples[: args.limit]

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]

    tasks = []
    for sample in samples:
        for seed in seeds:
            tasks.append((sample, seed))

    # Shuffle deterministically per process so workers start at different positions
    random.Random(args.seed + os.getpid()).shuffle(tasks)

    done = load_done_tasks(metadata_path) if args.resume else set()
    done_snapshot = load_done_snapshot(out_dir, worker_id) if args.resume else set()
    remaining = [t for t in tasks if task_id(t[0], t[1], args.instruction_field) not in done and task_id(t[0], t[1], args.instruction_field) not in done_snapshot]

    plan = {
        "model_dir": args.model_dir,
        "jsonl": args.jsonl,
        "out_dir": str(out_dir),
        "worker_id": worker_id,
        "device": device,
        "total_tasks": len(tasks),
        "done_tasks": len(done),
        "remaining_tasks": len(remaining),
        "config_fingerprint": config_fingerprint(args),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.dry_run:
        return

    pipe = load_pipeline(
        args.model_dir, device, args.vae_tiling, args.lora_scale,
        args.shift, args.mu_mode, args.sft_ckpt, args.sft_lora_ckpt
    )
    success = fail = 0
    start_time = time.time()

    for idx, (sample, seed) in enumerate(remaining):
        tid = task_id(sample, seed, args.instruction_field)
        lock_path, done_path = paths_for_key(out_dir, tid)

        if args.resume and is_done(done_path):
            continue
        if not try_acquire_lock(lock_path, args.lock_timeout):
            continue
        # A worker may have completed the task after our initial status check.
        if args.resume and is_done(done_path):
            release_lock(lock_path)
            continue

        instruction = sample[args.instruction_field]

        out_path = make_output_path(str(out_dir), sample["image_name"], instruction,
                                    seed=seed if len(seeds) > 1 else None)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        record = dict(sample)  # preserve all original fields
        record.update({
            "task_id": tid,
            "seed": seed,
            "output_image": out_path,
            "instruction_field": args.instruction_field,
            "status": "failed",
            "timestamp": time.time(),
        })

        try:
            input_image = Image.open(sample["image_name"]).convert("RGB")
            generator = torch.Generator(device).manual_seed(seed)
            t_gen0 = time.time()
            result = pipe(
                image=input_image,
                instruction=instruction,
                num_inference_steps=args.steps,
                mu_mode=args.mu_mode,
                guidance_scale=args.guidance,
                generator=generator,
                output_type="pil",
                target_pixels=args.target_pixels,
            )
            gen_time_s = round(time.time() - t_gen0, 1)
            output_image = result.images[0]
            output_image.save(out_path)

            record.update({
                "status": "success",
                "gen_time_s": gen_time_s,
                "steps": args.steps,
                "guidance": args.guidance,
                "shift": args.shift,
                "mu_mode": pipe.last_mu_mode,
                "mu": pipe.last_mu,
                "image_seq_len": pipe.last_image_seq_len,
                "target_pixels": args.target_pixels,
                "input_size": list(input_image.size),
                "output_size": list(output_image.size),
            })
            append_metadata(metadata_path, record)
            mark_done(done_path, {"task_id": tid, "status": "success"})
            success += 1
        except Exception as exc:
            record["error"] = str(exc)
            append_metadata(metadata_path, record)
            mark_done(done_path, {"task_id": tid, "status": "failed", "error": str(exc)})
            fail += 1
        finally:
            release_lock(lock_path)

        if (idx + 1) % 10 == 0:
            print(f"[{worker_id}] progress {idx+1}/{len(remaining)} "
                  f"success={success} fail={fail}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[{worker_id}] Done! success={success} fail={fail} "
          f"time={elapsed/60:.1f}min")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
