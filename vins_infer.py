#!/usr/bin/env python3
"""BluePixel image-editing inference (single GPU, FP32 and INT8 in one script).

The numerics mirror the reference release pipeline:

    denoising steps : [1000, 750, 500, 250] (4 steps)
    sigma shift     : mu_mode="training" (exponential, resolution only)
    sampler         : CPS (eta = 0.9) on top of FlowMatchEulerDiscreteScheduler
    guidance        : always disabled (guidance=None, matching the release config)
    components      : transformer fp32 (or int8 weight-only), VAE fp32,
                      text encoder bf16, latents bf16
    resolution      : aspect-ratio bucket, area 4096*4096, multiple 64
    ref latents     : AutoencoderDC (64x compression), scaling_factor norm

The transformer is loaded as int8 automatically when
``<model_dir>/transformer/config.json`` carries a ``quantization_config``
(written by the int8 weight-only quantizer). Otherwise it is loaded in fp32.

Usage:
    python vins_infer.py --model-dir /path/to/BluePixel-Turbo \
        --jsonl assets/test_examples.jsonl --out-dir ./output

    python vins_infer.py --model-dir /path/to/bluepixel_..._int8 \
        --base-model-dir /path/to/bluepixel_..._fp32 \
        --jsonl assets/test_examples.jsonl --out-dir ./output_int8
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import os.path as osp
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    Flux2Transformer2DModel,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

DEFAULT_BUCKET_RATIOS = (
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1",
    "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
)
DEFAULT_BUCKET_AREA = 4096 * 4096
DEFAULT_BUCKET_MULTIPLE = 64
DEFAULT_DENOISING_STEPS = [1000, 750, 500, 250]


# ---------------------------------------------------------------------------
# Resolution buckets
# ---------------------------------------------------------------------------
def parse_ratio(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid bucket ratio {value!r}; expected W:H")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bucket ratio {value!r}; values must be positive")
    divisor = math.gcd(width, height)
    return width // divisor, height // divisor


class ResolutionBucketSet:
    """Aligned target resolutions built with Klein's area convention."""

    def __init__(
        self,
        ratios=DEFAULT_BUCKET_RATIOS,
        *,
        target_area: int = DEFAULT_BUCKET_AREA,
        multiple: int = DEFAULT_BUCKET_MULTIPLE,
        min_side: int = 64,
    ) -> None:
        if target_area <= 0:
            raise ValueError("target_area must be positive")
        if multiple <= 0 or min_side <= 0:
            raise ValueError("multiple and min_side must be positive")
        self.target_area = int(target_area)
        self.multiple = int(multiple)
        self.min_side = max(int(min_side), self.multiple)
        self.ratio_pairs = tuple(parse_ratio(r) for r in ratios)
        if not self.ratio_pairs:
            raise ValueError("At least one resolution bucket is required")
        self.buckets = tuple(
            self._make_bucket(i, pair) for i, pair in enumerate(self.ratio_pairs)
        )

    def _make_bucket(self, bucket_id: int, pair: tuple[int, int]) -> dict[str, Any]:
        ratio_w, ratio_h = pair
        ratio = ratio_w / ratio_h
        raw_width = math.sqrt(self.target_area * ratio)
        raw_height = math.sqrt(self.target_area / ratio)
        width = max(self.min_side, int(raw_width) // self.multiple * self.multiple)
        height = max(self.min_side, int(raw_height) // self.multiple * self.multiple)
        while width * height > self.target_area and width > self.min_side:
            width -= self.multiple
        while width * height > self.target_area and height > self.min_side:
            height -= self.multiple
        return {
            "bucket_id": bucket_id,
            "ratio": f"{ratio_w}:{ratio_h}",
            "width": width,
            "height": height,
        }

    def __len__(self) -> int:
        return len(self.buckets)

    def __iter__(self):
        return iter(self.buckets)

    def __getitem__(self, bucket_id: int) -> dict[str, Any]:
        return self.buckets[bucket_id]

    def select(self, width: int, height: int) -> dict[str, Any]:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {width}x{height}")
        image_ratio = width / height
        return min(
            self.buckets,
            key=lambda b: abs(math.log(image_ratio) - math.log(b["width"] / b["height"])),
        )


# ---------------------------------------------------------------------------
# Image preprocessing / latents
# ---------------------------------------------------------------------------
def preprocess_ref_image_pil(image: Image.Image, target_height: int, target_width: int) -> torch.Tensor:
    """Resize+crop to (target_height, target_width) and rescale to [-1, 1]."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    src_w, src_h = image.size
    scale = max(target_width / src_w, target_height / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    image = image.crop((left, top, left + target_width, top + target_height))

    array = np.array(image, dtype="uint8", copy=True)
    array = torch.from_numpy(array).float() / 255.0
    array = array.permute(2, 0, 1).contiguous()
    return array * 2.0 - 1.0


def pack_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1).contiguous()


def unpack_flux2_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch_size, num_tokens, num_channels = tokens.shape
    if num_tokens != height * width:
        raise ValueError(
            f"Token count mismatch when unpacking latents: got {num_tokens}, expected {height * width}"
        )
    return tokens.permute(0, 2, 1).reshape(batch_size, num_channels, height, width).contiguous()


def _prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = prompt_embeds.shape
    out_ids = []
    for _ in range(batch_size):
        coords = torch.cartesian_prod(
            torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(seq_len)
        )
        out_ids.append(coords)
    return torch.stack(out_ids)


def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = latents.shape
    device = latents.device
    latent_ids = torch.cartesian_prod(
        torch.arange(1, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        torch.arange(1, device=device),
    )
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)


def _build_ref_image_ids_dcae(ref_latents: torch.Tensor, t_scale: int = 10, t_index: int = 0) -> torch.Tensor:
    batch_size, _, height, width = ref_latents.shape
    device = ref_latents.device
    t = torch.arange(1, device=device) * 0 + (t_scale + t_scale * t_index)
    coords = torch.cartesian_prod(
        t,
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        torch.arange(1, device=device),
    )
    return coords.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_prompts(
    prompts: list[str],
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    device: torch.device,
    max_sequence_length: int = 512,
    hidden_states_layers: tuple[int, int, int] = (9, 18, 27),
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    stacked = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    stacked = stacked.to(dtype=text_encoder.dtype, device=device)

    batch_size, num_layers, seq_len, hidden_dim = stacked.shape
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)
    text_ids = _prepare_text_ids(prompt_embeds).to(device)
    return prompt_embeds, text_ids


# ---------------------------------------------------------------------------
# Sigma schedule
# ---------------------------------------------------------------------------
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def compute_training_mu(image_seq_len: int, seq_len_cap: float = 4096.0) -> float:
    effective_seq_len = min(float(image_seq_len), seq_len_cap)
    return float(0.00016927 * effective_seq_len + 0.45666666)


def _time_shift_exponential(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _time_shift_linear(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return mu / (mu + (1 / t - 1) ** sigma)


def compute_sigmas(
    denoising_step_list: list[int],
    scheduler_config: Any,
    image_seq_len: int,
    num_steps: int | None = None,
    mu_mode: str = "training",
) -> list[float]:
    steps = denoising_step_list[:num_steps] if num_steps is not None else list(denoising_step_list)
    if not steps:
        raise ValueError("denoising_step_list must contain at least one step.")

    num_train_timesteps = int(getattr(scheduler_config, "num_train_timesteps", 1000))
    raw_sigmas = torch.tensor(
        [float(s) / float(num_train_timesteps) for s in steps], dtype=torch.float32
    )

    use_dynamic = bool(getattr(scheduler_config, "use_dynamic_shifting", False))
    if mu_mode == "training":
        mu = compute_training_mu(image_seq_len=image_seq_len)
    elif mu_mode == "empirical":
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=len(steps))
    elif use_dynamic:
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=len(steps))
    else:
        mu = None

    if mu is None:
        shift = float(getattr(scheduler_config, "shift", 1.0))
        shifted = shift * raw_sigmas / (1 + (shift - 1) * raw_sigmas)
    else:
        time_shift_type = str(getattr(scheduler_config, "time_shift_type", "exponential")).lower()
        if time_shift_type == "linear":
            shifted = _time_shift_linear(mu, 1.0, raw_sigmas)
        else:
            shifted = _time_shift_exponential(mu, 1.0, raw_sigmas)

    shift_terminal = getattr(scheduler_config, "shift_terminal", None)
    if shift_terminal:
        one_minus_z = 1 - shifted
        scale = one_minus_z[-1] / (1 - float(shift_terminal))
        shifted = 1 - (one_minus_z / scale)

    return [float(s.item()) for s in shifted]


# ---------------------------------------------------------------------------
# CPS scheduler
# ---------------------------------------------------------------------------
class CPSScheduler(FlowMatchEulerDiscreteScheduler):
    """Flow-matching CPS step with controllable stochasticity via eta."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eta: float = 0.0

    def set_eta(self, eta: float) -> None:
        self._eta = eta

    @property
    def eta(self) -> float:
        return self._eta

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None) -> None:
        if sigmas is not None:
            sigmas = torch.tensor(sigmas, dtype=torch.float32)
            if len(sigmas) == 0 or sigmas[-1] != 0.0:
                sigmas = torch.cat([sigmas, torch.zeros(1)])
            timesteps = sigmas[:-1] * self.config.num_train_timesteps
            self.timesteps = timesteps.to(device=device)
            self.sigmas = sigmas.to(device=device)
            self.num_inference_steps = len(self.timesteps)
            self._step_index = None
            self._begin_index = None
            return

        if mu is None:
            super().set_timesteps(num_inference_steps=num_inference_steps, device=device)
        else:
            try:
                super().set_timesteps(num_inference_steps=num_inference_steps, device=device, mu=mu)
            except TypeError:
                super().set_timesteps(num_inference_steps=num_inference_steps, device=device)

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        sample_f32 = sample.to(torch.float32)
        v_pred_f32 = model_output.to(torch.float32)

        x0 = sample_f32 - sigma * v_pred_f32
        eps = sample_f32 + (1.0 - sigma) * v_pred_f32

        if sigma_next == 0.0 or self._eta == 0.0:
            prev_sample = sample_f32 + (sigma_next - sigma) * v_pred_f32
        else:
            rho = sigma_next * math.sin(self._eta * math.pi / 2)
            eps_coeff = sigma_next * math.cos(self._eta * math.pi / 2)
            prev_mean = (1.0 - sigma_next) * x0 + eps_coeff * eps
            noise = torch.randn(
                sample.shape, generator=generator, device=sample.device, dtype=torch.float32
            )
            prev_sample = prev_mean + rho * noise

        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


# ---------------------------------------------------------------------------
# INT8 weight-only linear
# ---------------------------------------------------------------------------
class Int8Linear(nn.Module):
    """Weight-only int8 linear with per-output-channel scale."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype)
        y = F.linear(x, w)
        y = y * self.weight_scale.to(x.dtype)
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)
        return y


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def read_quantization_config(transformer_dir: str) -> dict | None:
    cfg_path = osp.join(transformer_dir, "config.json")
    if not osp.isfile(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("quantization_config")


def load_int8_transformer(transformer_dir: str, compute_dtype: torch.dtype) -> nn.Module:
    """Build the transformer with Int8Linear layers and load int8 weights."""
    with open(osp.join(transformer_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    qcfg = cfg.pop("quantization_config")
    qkeys = set(qcfg["quantized_keys"])
    suffix = qcfg.get("scale_suffix", "_scale")
    for key in [k for k in cfg if k.startswith("_")]:
        cfg.pop(key)

    with torch.device("meta"):
        model = Flux2Transformer2DModel.from_config(cfg)

    n_replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and f"{name}.weight" in qkeys:
            _set_submodule(
                model,
                name,
                Int8Linear(module.in_features, module.out_features, module.bias is not None),
            )
            n_replaced += 1
    if n_replaced != len(qkeys):
        raise RuntimeError(
            f"int8 layer mismatch: replaced {n_replaced} Linear modules, "
            f"but quantization_config lists {len(qkeys)} keys"
        )

    state = {}
    for shard in sorted(glob.glob(osp.join(transformer_dir, "*.safetensors"))):
        with safe_open(shard, "pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    missing_scales = [k for k in qkeys if (k + suffix) not in state]
    if missing_scales:
        raise RuntimeError(f"missing scale tensors for {len(missing_scales)} quantized weights")

    model.load_state_dict(state, strict=True, assign=True)

    leftover = [
        n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta
    ]
    if leftover:
        raise RuntimeError(f"tensors still on meta after load: {leftover[:5]}")

    for p in model.parameters():
        p.data = p.data.to(compute_dtype)

    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype {name!r}; choose from {sorted(DTYPE_MAP)}")
    return DTYPE_MAP[name]


def load_pipeline(
    model_dir: str,
    base_model_dir: str | None = None,
    device: str | torch.device = "cuda",
    dtype: str = "bf16",
    int8_compute_dtype: str = "fp32",
    vae_tiling: bool = False,
) -> "BluePixelPipeline":
    """Load a merged BluePixel model, auto-detecting int8 vs fp32 transformer."""
    if int8_compute_dtype not in ("fp32", "bf16"):
        raise ValueError(f"int8_compute_dtype must be fp32|bf16, got {int8_compute_dtype!r}")

    device = torch.device(device)
    component_dtype = _resolve_torch_dtype(dtype)
    base_dir = base_model_dir or model_dir
    transformer_dir = osp.join(model_dir, "transformer")
    if not osp.isdir(transformer_dir):
        raise FileNotFoundError(f"transformer/ not found under {model_dir}")

    qcfg = read_quantization_config(transformer_dir)
    is_int8 = qcfg is not None

    print(f"[Load] model_dir={model_dir}")
    print(f"[Load] base_model_dir={base_dir}")
    print(f"[Load] transformer={'int8' if is_int8 else 'fp32'}")

    if is_int8:
        transformer = load_int8_transformer(
            transformer_dir, DTYPE_MAP[int8_compute_dtype]
        )
        print(
            f"[Load] int8 transformer: {len(qcfg['quantized_keys'])} Int8Linear layers, "
            f"compute dtype={int8_compute_dtype}"
        )
    else:
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_dir, subfolder="transformer", torch_dtype=torch.float32
        )

    vae = AutoencoderDC.from_pretrained(base_dir, subfolder="vae", torch_dtype=torch.float32)
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        base_dir, subfolder="text_encoder", torch_dtype=component_dtype
    )
    tokenizer = Qwen2TokenizerFast.from_pretrained(base_dir, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_dir, subfolder="scheduler")

    for module in (vae, text_encoder, transformer):
        module.requires_grad_(False)
        module.eval()

    if vae_tiling:
        vae.enable_tiling()
        print("[Load] VAE tiling enabled")

    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    return BluePixelPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        weight_dtype=component_dtype,
        device=device,
        is_int8=is_int8,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class BluePixelPipeline:
    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        scheduler: FlowMatchEulerDiscreteScheduler,
        weight_dtype: torch.dtype,
        device: torch.device,
        is_int8: bool,
    ) -> None:
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.weight_dtype = weight_dtype
        self.device = device
        self.is_int8 = is_int8
        self.num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        self.x0_scheduler = CPSScheduler.from_config(scheduler.config)
        self.last_mu_mode: str | None = None
        self.last_image_seq_len: int | None = None

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.transformer.parameters()).dtype
        except StopIteration:
            return torch.float32

    @torch.no_grad()
    def _encode_ref_image(
        self,
        image_path: str,
        bucket_set: ResolutionBucketSet,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(image_path) as image:
            pil = image.convert("RGB")
        bucket = bucket_set.select(*pil.size)
        ref_tensor = preprocess_ref_image_pil(
            pil, target_height=bucket["height"], target_width=bucket["width"]
        )
        ref_pixels = ref_tensor.unsqueeze(0).to(device=self.vae.device, dtype=self.vae.dtype)

        latents = self.vae.encode(ref_pixels)[0]
        latents = latents * self.vae.config.scaling_factor
        ref_ids = _build_ref_image_ids_dcae(latents, t_scale=10, t_index=0)
        ref_tokens = pack_flux2_latents(latents)
        return ref_tokens, ref_ids

    @torch.no_grad()
    def _predict_noise(
        self,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        text_ids: torch.Tensor,
        ref_tokens: torch.Tensor,
        ref_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = noisy_latents.shape
        latent_ids = _prepare_latent_ids(noisy_latents)
        model_dtype = self._model_dtype()
        packed = pack_flux2_latents(noisy_latents).to(model_dtype)

        model_timestep = timesteps.to(device=noisy_latents.device, dtype=packed.dtype)
        model_timestep = model_timestep / float(self.num_train_timesteps)

        full_hidden = torch.cat(
            [packed, ref_tokens.to(device=noisy_latents.device, dtype=model_dtype)], dim=1
        )
        full_ids = torch.cat([latent_ids, ref_ids.to(device=noisy_latents.device)], dim=1)

        pred = self.transformer(
            hidden_states=full_hidden,
            timestep=model_timestep,
            guidance=None,
            encoder_hidden_states=prompt_embeds.to(
                device=noisy_latents.device, dtype=model_dtype
            ),
            txt_ids=text_ids.to(device=noisy_latents.device),
            img_ids=full_ids,
            return_dict=False,
        )[0]
        pred_tokens = pred[:, : packed.size(1), :]
        return unpack_flux2_latents(pred_tokens, height=height, width=width).to(
            noisy_latents.dtype
        )

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> list[Image.Image]:
        latents = (latents / self.vae.config.scaling_factor).to(
            device=self.vae.device, dtype=self.vae.dtype
        )
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        array = (
            (image * 255)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .permute(0, 2, 3, 1)
            .numpy()
        )
        return [Image.fromarray(x) for x in array]

    @torch.no_grad()
    def generate(
        self,
        image_path: str,
        instruction: str,
        seed: int = 42,
        num_steps: int = 4,
        denoising_step_list: list[int] | None = None,
        cps_eta: float = 0.9,
        mu_mode: str = "training",
        bucket_set: ResolutionBucketSet | None = None,
        sigma_image_seq_len: int | None = None,
    ) -> Image.Image:
        """Generate one edited image for a single (image, instruction) pair.

        Classifier-free guidance is intentionally not exposed: the released
        distilled checkpoints are evaluated with `guidance_scale=1.0`, i.e. a
        single conditional forward pass (`guidance=None`).
        """
        denoising_step_list = denoising_step_list or DEFAULT_DENOISING_STEPS
        bucket_set = bucket_set or ResolutionBucketSet()

        self.x0_scheduler.set_eta(cps_eta)

        prompt_embeds, text_ids = encode_prompts(
            prompts=[instruction],
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        ref_tokens, ref_ids = self._encode_ref_image(image_path, bucket_set)

        effective_seq_len = (
            sigma_image_seq_len if sigma_image_seq_len is not None else int(ref_ids.shape[1])
        )
        sigmas = compute_sigmas(
            denoising_step_list=denoising_step_list,
            scheduler_config=self.scheduler.config,
            image_seq_len=effective_seq_len,
            num_steps=min(num_steps, len(denoising_step_list)),
            mu_mode=mu_mode,
        )
        actual_steps = min(num_steps, len(sigmas))
        self.last_mu_mode = mu_mode
        self.last_image_seq_len = effective_seq_len

        self.x0_scheduler.set_timesteps(sigmas=sigmas)
        self.x0_scheduler._step_index = 0

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent_h = int(ref_ids[..., 1].max().item()) + 1
        latent_w = int(ref_ids[..., 2].max().item()) + 1
        latent_channels = self.transformer.config.in_channels
        x_t = torch.randn(
            1,
            latent_channels,
            latent_h,
            latent_w,
            device=self.device,
            dtype=self.weight_dtype,
            generator=generator,
        )
        x0 = x_t

        for idx in range(actual_steps):
            t = self.x0_scheduler.timesteps[idx]
            t_batch = t.expand(1).to(device=self.device, dtype=torch.float32)

            v_pred = self._predict_noise(
                x_t, prompt_embeds, t_batch, text_ids, ref_tokens, ref_ids
            )

            step_output = self.x0_scheduler.step(v_pred, t, x_t, return_dict=True)
            sigma = self.x0_scheduler.sigmas[idx]
            x0 = (x_t.float() - sigma * v_pred.float()).to(self.weight_dtype)
            x_t = step_output.prev_sample

        return self._decode(x0)[0]


# ---------------------------------------------------------------------------
# Dataset / output helpers
# ---------------------------------------------------------------------------
def load_jsonl_dataset(
    dataset_path: str,
    image_field: str = "image_name",
    prompt_field: str = "edit_prompt",
    task_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                samples.append(item)
    if task_types is not None:
        task_set = set(task_types)
        samples = [s for s in samples if s.get("task_type", "") in task_set]
    return samples


def _flatten_path(path: str) -> str:
    return path.lstrip("/").replace("/", "_")


def _is_complete(path: str) -> bool:
    return osp.isfile(path) and osp.getsize(path) > 0


def _atomic_save(image: Image.Image, path: str) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    fmt = "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
    kwargs = {"quality": 95} if fmt == "JPEG" else {}
    image.save(tmp, format=fmt, **kwargs)
    os.replace(tmp, path)


def _write_results_jsonl(dataset_path: str, seed_dir: str, image_field: str) -> str:
    results_path = osp.join(seed_dir, "results.jsonl")
    with open(dataset_path, "r", encoding="utf-8") as fin, open(
        results_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            flat_name = _flatten_path(obj.get(image_field, ""))
            obj["edit_image"] = osp.abspath(osp.join(seed_dir, flat_name))
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return results_path


# ---------------------------------------------------------------------------
# CLI
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BluePixel 4-step image editing inference (single GPU, fp32 / int8)"
    )
    p.add_argument("--model-dir", type=str, required=True,
                   help="Model directory; int8 is auto-detected from transformer/config.json")
    p.add_argument("--base-model-dir", type=str, default=None,
                   help="Optional directory holding vae/text_encoder/tokenizer/scheduler "
                        "(defaults to --model-dir)")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--image-field", type=str, default="image_name")
    p.add_argument("--instruction-field", type=str, default="edit_prompt")
    p.add_argument("--task-types", type=str, nargs="*", default=None)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seed list, e.g. 0,42,777")
    p.add_argument("--steps", type=positive_int, default=4)
    p.add_argument("--denoising-step-list", type=int, nargs="+", default=DEFAULT_DENOISING_STEPS)
    p.add_argument("--cps-eta", type=nonnegative_float, default=0.9)
    p.add_argument("--mu-mode", type=str, choices=["none", "empirical", "training"],
                   default="training")
    p.add_argument("--sigma-image-seq-len", type=int, default=None)
    p.add_argument("--dtype", type=str, choices=sorted(DTYPE_MAP), default="bf16",
                   help="Dtype for text encoder and latent storage")
    p.add_argument("--int8-compute-dtype", type=str, choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--bucket-ratios", type=str, nargs="+", default=list(DEFAULT_BUCKET_RATIOS))
    p.add_argument("--bucket-area", type=int, default=DEFAULT_BUCKET_AREA)
    p.add_argument("--bucket-multiple", type=int, default=DEFAULT_BUCKET_MULTIPLE)
    p.add_argument("--vae-tiling", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--no-results-jsonl", dest="results_jsonl", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [args.seed]
    )
    if not seeds:
        raise SystemExit("[ERROR] empty seed list")

    bucket_set = ResolutionBucketSet(
        args.bucket_ratios, target_area=args.bucket_area, multiple=args.bucket_multiple
    )

    samples = load_jsonl_dataset(
        args.jsonl, args.image_field, args.instruction_field, args.task_types
    )
    total = len(samples)
    start_idx = args.start_idx
    end_idx = min(start_idx + args.num_samples, total) if args.num_samples else total
    samples = samples[start_idx:end_idx]
    if args.limit:
        samples = samples[: args.limit]

    plan: dict[str, Any] = {
        "model_dir": args.model_dir,
        "base_model_dir": args.base_model_dir or args.model_dir,
        "dataset": args.jsonl,
        "eval_range": [start_idx, end_idx, total],
        "output_dir": args.out_dir,
        "seeds": seeds,
        "steps": args.steps,
        "denoising_step_list": args.denoising_step_list[: args.steps],
        "cps_eta": args.cps_eta,
        "mu_mode": args.mu_mode,
        "dtype": args.dtype,
        "int8_compute_dtype": args.int8_compute_dtype,
        "bucket_area": args.bucket_area,
        "bucket_multiple": args.bucket_multiple,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.dry_run:
        return

    pipe = load_pipeline(
        model_dir=args.model_dir,
        base_model_dir=args.base_model_dir,
        device=args.device,
        dtype=args.dtype,
        int8_compute_dtype=args.int8_compute_dtype,
        vae_tiling=args.vae_tiling,
    )

    total_fail = 0
    for seed in seeds:
        seed_dir = osp.join(args.out_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        items = [
            (sample, osp.join(seed_dir, _flatten_path(sample.get(args.image_field, ""))))
            for sample in samples
        ]
        pending = [
            (position, sample, path)
            for position, (sample, path) in enumerate(items)
            if not (args.resume and _is_complete(path))
        ]

        print(f"\n[seed {seed}] done={len(items) - len(pending)} pending={len(pending)} -> {seed_dir}")
        if not pending:
            if args.results_jsonl:
                path = _write_results_jsonl(args.jsonl, seed_dir, args.image_field)
                print(f"[seed {seed}] results.jsonl -> {path}")
            continue

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        success = fail = 0
        total_time = 0.0
        for batch_idx, (position, sample, out_path) in enumerate(pending):
            instruction = sample.get(args.instruction_field, sample.get("prompt", ""))
            image_path = sample.get(args.image_field, sample.get("image", ""))
            t0 = time.time()
            try:
                image = pipe.generate(
                    image_path=image_path,
                    instruction=instruction,
                    seed=seed + batch_idx,
                    num_steps=args.steps,
                    denoising_step_list=args.denoising_step_list,
                    cps_eta=args.cps_eta,
                    mu_mode=args.mu_mode,
                    bucket_set=bucket_set,
                    sigma_image_seq_len=args.sigma_image_seq_len,
                )
                _atomic_save(image, out_path)
                success += 1
                total_time += time.time() - t0
                print(
                    f"[seed {seed}] {position + 1}/{len(items)} "
                    f"{time.time() - t0:.2f}s -> {out_path}"
                )
            except Exception as exc:
                fail += 1
                print(f"[seed {seed}] {position + 1}/{len(items)} FAILED {image_path}: {exc}")

        if args.results_jsonl:
            path = _write_results_jsonl(args.jsonl, seed_dir, args.image_field)
            print(f"[seed {seed}] results.jsonl -> {path}")

        avg = total_time / max(success, 1)
        print(
            f"\n[seed {seed} Done] success={success} fail={fail} "
            f"avg={avg:.2f}s/sample -> {seed_dir}"
        )
        total_fail += fail

    if total_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
