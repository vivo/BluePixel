<div align="center">

# BluePixelModel

**原生 4K 图像编辑模型，基于 Flux 2 Klein 与 DCAE**

▶ [在线体验](https://bluepixel.vivo.com.cn/bluepixel-model) • 🤗 [Hugging Face](https://huggingface.co/openvivo/BluePixel-Turbo/) • [论文](https://arxiv.org/abs/2610.19882) • 📜 [Apache 2.0 License](LICENSE) • [English](README.md)

</div>

## 简介

我们很高兴开源 **BluePixelModel**，一个从 1K 预训练的 [Flux 2 Klein](https://blackforestlabs.ai/) 编辑器升级而来的**原生 4K 图像编辑模型**。

BluePixelModel 的四大核心特点：

- **原生 4K 直出** — 支持 4096×4096 方形输出，无需级联超分；非方形图片按目标像素预算近似保持原始宽高比。
- **4 步推理** — 通过源锚定目标投影（Source-Anchored Target Projection，SATP），从 50 步教师模型蒸馏得到。
- **高效 4K 训练** — 使用 DCAE-F64（f=64）将 4K 图像的 DiT token 数控制在 1K 级别，32 张 Kunlunxin XPU 上 4 天完成初始 4K 适配。
- **持续学习能力** — 通过 ACE-LoRA + OPD 吸收长尾垂类编辑技能，避免任务干扰与灾难性遗忘。

## 新闻

- 模型权重：已发布的 4 步模型见 [BluePixel-Turbo](https://huggingface.co/openvivo/BluePixel-Turbo/)。
- 2026.10.13: BluePixelModel 推理代码与模型文档开源。

## 快速开始

### 环境要求

```bash
pip install -r requirements.txt
```

运行示例前，下载完整模型：

```bash
hf download openvivo/BluePixel-Turbo --local-dir ./BluePixel-Turbo
```

使用支持 BF16 的 CUDA 环境。依赖范围对应 Diffusers 0.36 API，实际 GPU 推理需在部署环境验证。

### 单图编辑

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

### 批量推理

准备一个 jsonl 文件，每行一条任务，包含 `image_name`（输入图像路径）和 `edit_prompt`（编辑指令）字段：

```json
{"image_name": "/path/to/input.png", "edit_prompt": "make the hair red"}
```

运行：

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

默认跳过成功任务，失败任务会重试。`--no-resume` 重新执行全部选中任务并覆盖对应输出图片，请使用单个 worker。切换模型或推理参数时，请使用新的输出目录。

### 推荐分辨率

BluePixelModel 原生支持 4K，可通过 `--target-pixels` 控制输出像素数：

```python
target_pixels = {
    "1K": 1024 * 1024,      # 1048576
    "2K": 1440 * 1440,      # 2073600
    "4K": 4096 * 4096,      # 16777216
}
```

### 默认参数

| 参数 | 默认值 | 说明 |
|------|------|------|
| `num_inference_steps` | 4 | 去噪步数 |
| `guidance_scale` | 3.0 | CFG 引导强度 |
| `target_pixels` | 16777216 | 目标像素数（4K = 4096×4096） |
| `mu_mode` | `"empirical"` | μ-shift 模式 |

### 显存优化

对于显存受限的设备，可以启用 VAE tiling：

```python
pipe = load_pipeline(
    "BluePixel-Turbo", device="cuda", vae_tiling=True, mu_mode="empirical"
)
```

或在命令行添加 `--vae-tiling`。

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-dir` | str | **必填** | 模型权重目录 |
| `--jsonl` | str | **必填** | 任务列表 jsonl 文件 |
| `--out-dir` | str | 自动推导 | 输出目录 |
| `--target-pixels` | int | `16777216` | 目标像素数（4K = 4096×4096） |
| `--steps` | int | `4` | 去噪步数 |
| `--guidance` | float | `3.0` | CFG 引导强度 |
| `--seed` | int | `42` | 随机种子 |
| `--seeds` | str | `None` | 逗号分隔的多个随机种子 |
| `--shift` | float | `0.0` | DiT shift 参数 |
| `--mu-mode` | str | `empirical` | μ-shift 模式（`empirical` / `training`） |
| `--instruction-field` | str | `edit_prompt` | 编辑指令字段名 |
| `--device-id` | int | `0` | CUDA device id（通常通过 `CUDA_VISIBLE_DEVICES` 控制） |
| `--lora-scale` | float | `None` | LoRA 缩放系数 |
| `--sft-ckpt` | str | `None` | SFT 全量模型 checkpoint 路径 |
| `--sft-lora-ckpt` | str | `None` | SFT LoRA checkpoint 路径 |
| `--vae-tiling` | flag | `False` | 启用 VAE tiling 以节省显存 |
| `--limit` | int | `None` | 仅处理前 N 条任务 |
| `--lock-timeout` | int | `1800` | 文件锁超时时间（秒） |
| `--metadata-file` | str | `metadata.jsonl` | 输出元数据文件名 |
| `--no-resume` | flag | `False` | 重跑选中任务并覆盖输出（单 worker） |
| `--dry-run` | flag | `False` | 只打印任务计划，不执行推理 |
| `--check` | flag | `False` | 仅检查任务状态，不执行推理 |

## 训练

本模型基于 [Flux 2 Klein](https://blackforestlabs.ai/) 进行训练，整体流程分为四个阶段：

| 阶段 | 目标 | 关键方法 |
| :--- | :--- | :--- |
| **Preparation** | 4K 分辨率适配 | DCAE-F64 + 渐进式压缩迁移（f=8 → 32 → 64） |
| **Enhancement SFT** | 编辑质量提升 | 渐进分辨率 SFT + Benchmark-driven 微调 + ACE-OPD |
| **RL** | 美学与指令对齐 | DiffusionNFT 两阶段优化 |
| **Distillation** | 加速推理 | Source-Anchored Target Projection (SATP) |

### Preparation：4K 分辨率适配

将 Flux 2 Klein 的原始 f=8 VAE 替换为 DCAE-F64（f=64），按压缩率 f=8 → 32 → 64 渐进迁移，每级执行三个步骤：

| 子阶段 | 训练内容 | 损失函数 |
| :--- | :--- | :--- |
| 输入投影蒸馏 | 训练新的输入投影层，对齐新旧 VAE 编码 | 双 VAE 对齐损失 |
| 接口校准 | 冻结 DiT backbone，仅训练输入 / 输出投影 | Flow Matching Loss |
| Backbone 适配 | 轻量微调 DiT backbone | Flow Matching Loss |

- 约 **200K 样本**，在 **32 张 Kunlunxin XPU** 上训练约 **4 天**。

### Enhancement SFT：增强监督微调

- **渐进式质量 SFT**：先在 1024×1024 上训练，再迁移到原生 4096×4096；
- **Benchmark-driven 微调**：数据重平衡 + 多 epoch model merging；
- **Continual Learning（ACE-OPD）**：用 ACE-LoRA 分组学习长尾技能，每组通过 OPD 蒸馏回基础模型，逐组累积能力。

### RL：强化学习对齐

基于 DiffusionNFT 分两阶段优化：

| 阶段 | 任务 | Reward |
| :--- | :--- | :--- |
| 第一阶段 | T2I 文本到图像 | PickScore（美学优化） |
| 第二阶段 | I2I 图像编辑 | BluePixReward（指令保真度优化） |

### Distillation：推理步数蒸馏

通过源锚定目标投影（Source-Anchored Target Projection，SATP）将 50 步教师模型蒸馏为 **4 步学生模型**：

- 非编辑区域锚定到 source latent，保证背景一致性；
- 4K 编辑在单张 Kunlunxin P800 XPU 上仅需约 **8.7 秒**；
- 相比 50 步教师实现约 **4× 推理加速**。

## 模型架构

### 整体配置

| 组件 | 配置 |
| :--- | :--- |
| **Base Model** | [FLUX.2-Klein](https://blackforestlabs.ai/) — 1K 预训练图像编辑器 |
| **DiT** | 9B 双流 + 单流 Transformer |
| **Text Encoder** | Qwen3 |
| **VAE（原始）** | f=8，2×2 patchify，32 通道 |
| **VAE（ ours ）** | **DCAE-F64**：f=64，1×1 patchify，128 通道 |
| **输出分辨率** | 原生 4096×4096 |
| **推理步数** | 4 步（学生模型） |

### 4K Token 控制

| 输入分辨率 | VAE | patchify | Latent 尺寸 | Token 数 |
| :--- | :--- | :--- | :--- | :--- |
| 1024×1024 | f=8，2×2 | 8× | 128×128 | ~4096 |
| 4096×4096 | f=8，2×2 | 8× | 512×512 | ~65536 |
| **4096×4096** | **f=64，1×1** | **64×** | **64×64** | **~4096** |

> 4K 场景下，DCAE-F64 将 token 数从 **65536** 降到约 **4096**，与 1K 编辑的计算量相当。

### 推理 Pipeline

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

## 项目结构

```
BluePixelModel/
├── vins_infer.py                     # 批量推理脚本（支持断点续传、多进程文件锁）
├── run_vins_bench_v2.sh              # 单卡推理启动脚本（一键测试）
├── assets/
│   ├── test_input.jpg                # 夜景人像（毛衣改色）
│   ├── test_hiking.jpg               # 户外人像（外套改色）
│   ├── test_forest.jpg               # 林间人像（帽子改色）
│   └── test_examples.jsonl           # 示例任务列表（3 张照片、3 条编辑指令）
├── README.md
├── README_CN.md
└── LICENSE
```

## 评测结果

### Image Editing Benchmarks

| Method | Resolution | Metric | Score |
|--------|-----------|--------|-------|
| BluePixelModel (50-step) | 4K | - | - |
| BluePixelModel-Distill (4-step) | 4K | - | - |

## 声明、协议、引用

### 声明

我们在此郑重声明，对于所有使用开源模型的有关方，强烈呼吁不要进行任何损害国家社会安全或违反相关法律的行为，也恳请使用者不把 BluePixelModel 用于未经适当安全审批和备案的产品应用中。请务必在合法、合规的前提下开展一切业务活动，我们期望所有使用者都能以此为准。

同时，本模型"按原样"提供，我们也已尽全力确保数据的合规性，但由于模型训练和数据的复杂性，仍可能存在一些无法预估的问题，我们也强烈建议使用者对模型应用风险做详尽评估，确保应用的合法合规，如使用 BluePixelModel 开源模型而导致的任何问题，我们将不承担任何责任。

### 协议

本项目基于 [Apache 2.0 License](LICENSE) 开源。

相关依赖组件及其许可证信息请参阅 [NOTICE](NOTICE) 文件。

### 引用

```
@misc{2026bluepixelmodel,
    title={BluePixelModel: 4K Resolution Image Editing via Flux 2 Klein with DCAE},
    author={BluePixelModel Team},
    howpublished = {\url{https://github.com/vivo/BluePixel}},
    year={2026}
}
```

### 致谢

- [Black Forest Labs](https://blackforestlabs.ai/) — Flux 2 Klein 基座模型
- [SANA / NVlabs](https://github.com/NVlabs/SANA) — DCAE 架构
- [Qwen Team](https://qwenlm.github.io/) — Qwen3 文本编码器
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) — 推理框架基础
