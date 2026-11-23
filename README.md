<div align="center">

# BluePixelModel

**Native 4K Image Editing Model based on Flux 2 Klein and DCAE**

🌐 [Project Page](https://tech.vivo.com/BluePixel/) • 🎨 [Demo](https://bluepixel.vivo.com.cn/bluepixel-model) • 🤗 [Turbo](https://huggingface.co/openvivo/BluePixel-Turbo/) / [Turbo-INT8](https://huggingface.co/openvivo/BluePixel-Turbo-INT8) • 📄 [Paper](https://arxiv.org/abs/2610.19882) • 🌍 [中文](README_CN.md)

</div>

## Updates

2026.11.17 — Released [BluePixel-Turbo-INT8](https://huggingface.co/openvivo/BluePixel-Turbo-INT8), a quantized version with only a slight loss in quality compared to BluePixel-Turbo. We also released the [ComfyUI node](https://registry.comfy.org/nodes/bluepixel-comfyui), which supports both BluePixel-Turbo and BluePixel-Turbo-INT8. Thanks to the ModelScope community for making both models available: [BluePixel-Turbo](https://www.modelscope.cn/models/vivo/BluePixel-Turbo) and [BluePixel-Turbo-INT8](https://www.modelscope.cn/models/vivo/BluePixel-Turbo-INT8).

2026.11.03 — We update BluePixel to the **SATP-distilled version**, leveraging teacher–student distillation to enable high-quality image editing with significantly fewer sampling steps and lower inference costs. This update is an intermediate checkpoint without additional post-training. Further optimization, including reinforcement learning, is planned to improve aesthetic quality and instruction-following fidelity. 🤗 [Hugging Face](https://huggingface.co/openvivo/BluePixel-Turbo/)

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

### ComfyUI Workflows

Install the [BluePixel ComfyUI node](https://registry.comfy.org/nodes/bluepixel-comfyui), then download the appropriate workflow and drag it into ComfyUI:

- [BluePixel-Turbo 4K editing workflow](workflows/bluepixel_edit_4k.json)
- [BluePixel-Turbo-INT8 4K editing workflow](workflows/bluepixel_edit_4k_int8.json)

Prepare the models according to the node's instructions, select the model in the loader node, and choose an input image in the `Load Image` node. The INT8 workflow also requires the corresponding BluePixel-Turbo components.

### Requirements

```bash
pip install -r requirements.txt
```

Download the complete checkpoint before running the examples:

```bash
hf download openvivo/BluePixel-Turbo --local-dir ./BluePixel-Turbo
```

Use a CUDA environment with BF16 support. The dependencies include `diffusers>=0.40.0`; see `requirements.txt` for the full list.

### Single Image Editing

```python
from vins_infer import load_pipeline

pipe = load_pipeline(
    "BluePixel-Turbo",
    device="cuda",
    vae_tiling=False,
)

result = pipe.generate(
    image_path="assets/test_input.jpg",
    instruction="Change the gray sweater to burgundy red.",
    seed=42,
    num_steps=4,
    cps_eta=0.9,
    mu_mode="training",
)

result.save("output.jpg")
```

### INT8 Model

The INT8 release uses the same script: when `transformer/config.json` carries a
`quantization_config`, the transformer is loaded as weight-only int8
(per-output-channel scale); otherwise FP32 is used. An INT8 directory only holds
the transformer, so point `--base-model-dir` at the FP32 release for
vae / text_encoder / tokenizer / scheduler.

```bash
CUDA_VISIBLE_DEVICES=0 python vins_infer.py \
    --model-dir /path/to/bluepixel_..._int8 \
    --base-model-dir /path/to/BluePixel-Turbo \
    --jsonl /path/to/your_tasks.jsonl \
    --out-dir ./output_int8 \
    --bucket-area 16777216 \
    --int8-compute-dtype fp32
```

### Batch Inference

Prepare a jsonl file with one task per line, each containing `image_name` (input image path) and `edit_prompt` (editing instruction) fields:

```json
{"image_name": "/path/to/input.png", "edit_prompt": "make the hair red"}
```

Run directly:

```bash
CUDA_VISIBLE_DEVICES=0 python vins_infer.py \
    --model-dir BluePixel-Turbo \
    --jsonl /path/to/your_tasks.jsonl \
    --out-dir ./output \
    --bucket-area 16777216 \
    --steps 4 \
    --cps-eta 0.9 \
    --mu-mode training \
    --seed 42 \
    --instruction-field edit_prompt
```

Alternatively, run the same task through the launch script. Its default model path is a placeholder, so set `MODEL_DIR`:

```bash
MODEL_DIR=BluePixel-Turbo JSONL=/path/to/your_tasks.jsonl OUT_DIR=./output \
    CUDA_VISIBLE_DEVICES=0 bash run_infer.sh
```

Outputs go to `--out-dir/seed{N}/`; a `results.jsonl` is written with the original
jsonl rows plus an `edit_image` field. `--resume` (default) skips existing output
files, `--no-resume` regenerates and overwrites them.

### Recommended Resolutions

BluePixel natively supports 4K. Control output pixel count via `--bucket-area`.
The bucket is selected by aspect ratio and aligned to `--bucket-multiple 64`
before resize+center-crop, matching the reference release pipeline.

```python
bucket_area = {
    "1K": 1024 * 1024,      # 1048576
    "2K": 1440 * 1440,      # 2073600
    "4K": 4096 * 4096,      # 16777216 (default)
}
```

### Default Parameters

| Parameter | Default | Description |
|------|------|------|
| `--steps` | 4 | Number of denoising steps |
| `--denoising-step-list` | `1000 750 500 250` | Custom denoising schedule |
| `--cps-eta` | 0.9 | CPS sampler stochasticity (0 = deterministic Euler) |
| `--bucket-area` | 16777216 | Target pixel count (4K = 4096×4096) |
| `--mu-mode` | `training` | Sigma shift mode (`none` / `empirical` / `training`) |
| `--dtype` | bf16 | Text encoder and latent dtype |
| `--int8-compute-dtype` | fp32 | INT8 forward compute dtype (`fp32` / `bf16`) |

### Memory Optimization

For devices with limited memory, add `--vae-tiling` in the command line.

The FP32 transformer needs ~36 GB of resident weights (36.3 GB transformer + 2.7 GB VAE
+ 8.2 GB bf16 text encoder), so 4K inference is best on 80 GB+ GPUs. The INT8
transformer weights are ~9.1 GB.

### Command-line Arguments

| Argument | Type | Default | Description |
|------|------|--------|------|
| `--model-dir` | str | **Required** | Model weights directory (INT8 auto-detected) |
| `--base-model-dir` | str | `None` | Directory with vae/text_encoder/tokenizer/scheduler (defaults to `--model-dir`) |
| `--jsonl` | str | **Required** | Task list jsonl file |
| `--out-dir` | str | **Required** | Output directory (per-seed subdirectories) |
| `--device` | str | `cuda` | Inference device |
| `--image-field` | str | `image_name` | Input image path field name |
| `--instruction-field` | str | `edit_prompt` | Editing instruction field name |
| `--task-types` | list | `None` | Only process the given task types |
| `--start-idx` | int | `0` | Dataset start index |
| `--num-samples` | int | `None` | Number of samples to process |
| `--limit` | int | `None` | Only process the first N tasks |
| `--seed` | int | `42` | Random seed |
| `--seeds` | str | `None` | Comma-separated multiple random seeds |
| `--steps` | int | `4` | Number of denoising steps |
| `--denoising-step-list` | list | `1000 750 500 250` | Denoising schedule |
| `--cps-eta` | float | `0.9` | CPS sampler stochasticity |
| `--mu-mode` | str | `training` | Sigma shift mode (`none` / `empirical` / `training`) |
| `--sigma-image-seq-len` | int | `None` | Override sequence length used for sigma shift |
| `--dtype` | str | `bf16` | Text encoder / latent dtype |
| `--int8-compute-dtype` | str | `fp32` | INT8 forward compute dtype |
| `--bucket-ratios` | list | 14 defaults | Bucket aspect-ratio set |
| `--bucket-area` | int | `16777216` | Bucket target area (4K) |
| `--bucket-multiple` | int | `64` | Bucket side alignment |
| `--vae-tiling` | flag | `False` | Enable VAE tiling to save memory |
| `--no-resume` | flag | `False` | Regenerate all selected tasks and overwrite outputs |
| `--no-results-jsonl` | flag | `False` | Skip writing results.jsonl |
| `--dry-run` | flag | `False` | Print task plan only, do not run inference |

## Training

The model is trained based on [Flux 2 Klein](https://blackforestlabs.ai/). The overall process is divided into four stages:

| Stage | Objective | Key Method |
| :--- | :--- | :--- |
| **Preparation** | 4K resolution adaptation | DCAE-F64 + progressive compression migration (f=8 → 32 → 64) |
| **Enhancement SFT** | Editing quality improvement | Progressive resolution SFT + benchmark-driven tuning + ACE-OPD |
| **RL** | Aesthetic and instruction alignment | Two-stage DiffusionNFT optimization |
| **Distillation** | Faster inference | Source-Anchored Target Projection (SATP) |

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

Compress the 50-step teacher into a **4-step student** via source-anchored target projection (SATP):

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
             │ CPS (eta=0.9)       │
             │ [Noisy, Source Img]  │
             │ Joint Attention      │
             │ No CFG (scale=1.0)   │
             │ Training μ-shift     │
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
├── vins_infer.py                     # Single-GPU inference (FP32 / INT8, resume support)
├── run_infer.sh                      # Single-GPU launch script (one-command test)
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
