<div align="center">

# BluePixelModel

**Native 4K Image Editing Model based on Flux 2 Klein and DCAE**

▶ [Demo](https://bluepixel.vivo.com.cn/bluepixel-model) • 🤗 [Hugging Face](https://huggingface.co/openvivo/BluePixel-Turbo/) • [Paper](https://arxiv.org/abs/2610.19882) • 📜 [Apache 2.0 License](LICENSE) • [中文](README_CN.md)

</div>

## Introduction

We are excited to open-source **BluePixelModel**, a **native 4K image editing model** upgraded from a 1K-pretrained [Flux 2 Klein](https://blackforestlabs.ai/) editor.

Four key features of BluePixelModel:

- **Native 4K Direct Output** — Supports 4096×4096 square output without cascaded super-resolution. Non-square images approximately retain their aspect ratio within the requested pixel budget.
- **4-step inference** through source-anchored target projection (SATP), distilled from a 50-step teacher.
- **Efficient 4K Training** — Uses DCAE-F64 (f=64) to keep the 4K DiT token count at the 1K level, completing initial 4K adaptation on 32 Kunlunxin XPUs in about 4 days.
- **Continual Learning** — Absorbs long-tail vertical editing skills through ACE-LoRA + OPD, avoiding task interference and catastrophic forgetting.

## News

- Model weights: [BluePixel-Turbo](https://huggingface.co/openvivo/BluePixel-Turbo/) is the released 4-step checkpoint.
- 2026.10.13: BluePixelModel inference code and documentation are open-sourced.

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

Download the complete checkpoint before running the examples:

```bash
hf download openvivo/BluePixel-Turbo --local-dir ./BluePixel-Turbo
```

Use a CUDA environment with BF16 support. The dependencies target the Diffusers 0.36 API; GPU inference must be validated in your deployment environment.

### Single Image Editing

```python
import torch
from PIL import Image
from vins_infer import load_pipeline

pipe = load_pipeline(
    "BluePixel-Turbo",
    device="cuda",
    vae_tiling=False,
    lora_scale=None,
    shift=0,
    mu_mode="empirical",
)

result = pipe(
    image=Image.open("assets/test_input.jpg").convert("RGB"),
    instruction="Change the gray sweater to burgundy red.",
    num_inference_steps=4,
    guidance_scale=3.0,
    target_pixels=4096 * 4096,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

result.save("output.jpg")
```

### Batch Inference

Prepare a jsonl file with one task per line, each containing `image_name` (input image path) and `edit_prompt` (editing instruction) fields:

```json
{"image_name": "/path/to/input.png", "edit_prompt": "make the hair red"}
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python vins_infer.py \
    --model-dir BluePixel-Turbo \
    --jsonl /path/to/your_tasks.jsonl \
    --out-dir ./output \
    --target-pixels 16777216 \
    --steps 4 \
    --guidance 3.0 \
    --seed 42 \
    --mu-mode empirical \
    --instruction-field edit_prompt
```

Resume is enabled by default: successful tasks are skipped and failed tasks are retried. `--no-resume` reruns all selected tasks and overwrites their output images; use it with a single worker. Use a new output directory when changing the model or inference settings.

### Recommended Resolutions

BluePixelModel natively supports 4K. Control output pixel count via `--target-pixels`:

```python
target_pixels = {
    "1K": 1024 * 1024,      # 1048576
    "2K": 1440 * 1440,      # 2073600
    "4K": 4096 * 4096,      # 16777216
}
```

### Default Parameters

| Parameter | Default | Description |
|------|------|------|
| `num_inference_steps` | 4 | Number of denoising steps |
| `guidance_scale` | 3.0 | CFG guidance scale |
| `target_pixels` | 16777216 | Target pixel count (4K = 4096×4096) |
| `mu_mode` | `"empirical"` | μ-shift mode |

### Memory Optimization

For devices with limited memory, enable VAE tiling:

```python
pipe = load_pipeline(
    "BluePixel-Turbo", device="cuda", vae_tiling=True, mu_mode="empirical"
)
```

Or add `--vae-tiling` in the command line.

### Command-line Arguments

| Argument | Type | Default | Description |
|------|------|--------|------|
| `--model-dir` | str | **Required** | Path to model weights directory |
| `--jsonl` | str | **Required** | Task list jsonl file |
| `--out-dir` | str | Auto-derived | Output directory |
| `--target-pixels` | int | `16777216` | Target pixel count (4K = 4096×4096) |
| `--steps` | int | `4` | Number of denoising steps |
| `--guidance` | float | `3.0` | CFG guidance scale |
| `--seed` | int | `42` | Random seed |
| `--seeds` | str | `None` | Comma-separated multiple random seeds |
| `--shift` | float | `0.0` | DiT shift parameter |
| `--mu-mode` | str | `empirical` | μ-shift mode (`empirical` / `training`) |
| `--instruction-field` | str | `edit_prompt` | Name of the editing instruction field |
| `--device-id` | int | `0` | CUDA device id (usually set via `CUDA_VISIBLE_DEVICES`) |
| `--lora-scale` | float | `None` | LoRA scale factor |
| `--sft-ckpt` | str | `None` | SFT full model checkpoint path |
| `--sft-lora-ckpt` | str | `None` | SFT LoRA checkpoint path |
| `--vae-tiling` | flag | `False` | Enable VAE tiling to save memory |
| `--limit` | int | `None` | Only process the first N tasks |
| `--lock-timeout` | int | `1800` | File lock timeout in seconds |
| `--metadata-file` | str | `metadata.jsonl` | Output metadata file name |
| `--no-resume` | flag | `False` | Rerun selected tasks and overwrite outputs (single worker) |
| `--dry-run` | flag | `False` | Print task plan only, do not run inference |
| `--check` | flag | `False` | Only check task status, do not run inference |

## Training

The model is trained based on [Flux 2 Klein](https://blackforestlabs.ai/). The overall process is divided into four stages:

| Stage | Objective | Key Method |
| :--- | :--- | :--- |
| **Preparation** | 4K resolution adaptation | DCAE-F64 + progressive compression migration (f=8 → 32 → 64) |
| **Enhancement SFT** | Editing quality improvement | Progressive resolution SFT + benchmark-driven tuning + ACE-OPD |
| **RL** | Aesthetic and instruction alignment | Two-stage DiffusionNFT optimization |
| **Distillation** | Faster inference | Background-Anchored Step Distillation |

### Preparation: 4K Resolution Adaptation

Replace Flux 2 Klein's original f=8 VAE with DCAE-F64 (f=64), and migrate the compression ratio progressively f=8 → 32 → 64. At each level, three steps are performed:

| Sub-stage | Trainable Content | Loss Function |
| :--- | :--- | :--- |
| Input projection distillation | Train new input projection layer to align old and new VAE encodings | Dual VAE alignment loss |
| Interface calibration | Freeze DiT backbone; train only input / output projections | Flow Matching Loss |
| Backbone adaptation | Lightly adapt DiT backbone | Flow Matching Loss |

- Approximately **200K samples**, trained on **32 Kunlunxin XPUs** for about **4 days**.

### Enhancement SFT

- **Progressive quality SFT**: first train at 1024×1024, then transfer to native 4096×4096;
- **Benchmark-driven fine-tuning**: data rebalancing + multi-epoch model merging;
- **Continual Learning (ACE-OPD)**: use ACE-LoRA to learn long-tail skills in groups; each group is distilled back into the base model via OPD, accumulating capabilities group by group.

### RL: Reinforcement Learning Alignment

Optimize in two stages based on **DiffusionNFT**:

| Stage | Task | Reward |
| :--- | :--- | :--- |
| Stage 1 | T2I text-to-image | PickScore (aesthetic optimization) |
| Stage 2 | I2I image editing | BluePixReward (instruction fidelity optimization) |

### Distillation: Inference Step Distillation

Compress the 50-step teacher into a **4-step student** via background-anchored step distillation:

- Anchor non-edited regions to the source latent to ensure background consistency;
- 4K editing takes only about **8.7 seconds** on a single Kunlunxin P800 XPU;
- Achieves approximately **4× inference speedup** compared to the 50-step teacher.

## Model Architecture

### Overall Configuration

| Component | Configuration |
| :--- | :--- |
| **Base Model** | [FLUX.2-Klein](https://blackforestlabs.ai/) — 1K pretrained image editor |
| **DiT** | 9B dual-stream + single-stream Transformer |
| **Text Encoder** | Qwen3 |
| **VAE (Original)** | f=8, 2×2 patchify, 32 channels |
| **VAE (Ours)** | **DCAE-F64**: f=64, 1×1 patchify, 128 channels |
| **Output Resolution** | Native 4096×4096 |
| **Inference Steps** | 4 steps (student model) |

### 4K Token Control

| Input Resolution | VAE | Patchify | Latent Size | Token Count |
| :--- | :--- | :--- | :--- | :--- |
| 1024×1024 | f=8, 2×2 | 8× | 128×128 | ~4096 |
| 4096×4096 | f=8, 2×2 | 8× | 512×512 | ~65536 |
| **4096×4096** | **f=64, 1×1** | **64×** | **64×64** | **~4096** |

> At 4K, DCAE-F64 reduces token count from **65536** to about **4096**, comparable to 1K editing.

### Inference Pipeline

```
Input Image (4096×4096) + Edit Prompt
         │
         ▼
   ┌─────────────────┐     ┌────────────────┐
   │  DCAE-F64 Encode │     │  Qwen3 Encode  │
   │    (64× ↓)       │     │  (L9/L18/L27)  │
   └───────┬─────────┘     └───────┬────────┘
           │                       │
           ▼                       ▼
     Image Latents (64×64)   Text Embeds
           │                       │
           └──────────┬────────────┘
                      ▼
             ┌─ 4-step Denoising ─┐
             │ FlowMatchEuler       │
             │ [Noisy, Source Img]  │
             │ Joint Attention      │
             │ CFG (scale=3.0)      │
             │ Empirical μ-shift    │
             └──────────┬───────────┘
                        ▼
             DCAE-F64 Decode (64× ↑)
                        │
                        ▼
              Output Image (4096×4096)
```

## Project Structure

```
BluePixelModel/
├── vins_infer.py                     # Batch inference script (resume + file-lock multi-process)
├── run_vins_bench_v2.sh              # Single-GPU launch script (one-command test)
├── assets/
│   ├── test_input.jpg                # Night portrait (sweater color editing)
│   ├── test_hiking.jpg               # Hiking portrait (jacket color editing)
│   ├── test_forest.jpg               # Forest portrait (hat color editing)
│   └── test_examples.jsonl           # Example task list (3 photos, 3 editing instructions)
├── README.md
├── README_CN.md
└── LICENSE
```

## Evaluation

### Image Editing Benchmarks

| Method | Resolution | Metric | Score |
|--------|-----------|--------|-------|
| BluePixelModel (50-step) | 4K | - | - |
| BluePixelModel-Distill (4-step) | 4K | - | - |

## Disclaimer, License & Citation

### Disclaimer

We hereby solemnly declare that all parties using the open-source model are strongly urged not to engage in any behavior that harms national or social security or violates applicable laws. We also request that users do not apply BluePixelModel in products without proper security review and regulatory filing. Please ensure that all business activities are conducted lawfully and in compliance with all applicable regulations.

The model is provided "as is". While we have made every effort to ensure data compliance, the complexity of model training and data may still lead to unforeseen issues. We strongly recommend that users conduct thorough risk assessments to ensure lawful and compliant use. We assume no responsibility for any issues arising from the use of the BluePixelModel open-source model.

### License

This project is licensed under the [Apache 2.0 License](LICENSE).

For information about third-party dependencies and their licenses, please refer to the [NOTICE](NOTICE) file.

### Citation

```
@misc{2026bluepixelmodel,
    title={BluePixelModel: 4K Resolution Image Editing via Flux 2 Klein with DCAE},
    author={BluePixelModel Team},
    howpublished = {\url{https://github.com/vivo/BluePixel}},
    year={2026}
}
```

### Acknowledgements

- [Black Forest Labs](https://blackforestlabs.ai/) — Flux 2 Klein base model
- [SANA / NVlabs](https://github.com/NVlabs/SANA) — DCAE architecture
- [Qwen Team](https://qwenlm.github.io/) — Qwen3 text encoder
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) — Inference framework
