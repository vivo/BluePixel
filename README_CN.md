<div align="center">

# BluePixelModel

**原生 4K 图像编辑模型，基于 Flux 2 Klein 与 DCAE**

🌐 [项目主页](https://tech.vivo.com/BluePixel/) • 🎨 [在线体验](https://bluepixel.vivo.com.cn/bluepixel-model) • 🤗 [Turbo](https://huggingface.co/openvivo/BluePixel-Turbo/) / [Turbo-INT8](https://huggingface.co/openvivo/BluePixel-Turbo-INT8) • 📄 [论文](https://arxiv.org/abs/2610.19882) • 🌍 [English](README.md)

</div>

## 更新

2026.11.17 — 发布 [BluePixel-Turbo-INT8](https://huggingface.co/openvivo/BluePixel-Turbo-INT8) 量化版本，相比 BluePixel-Turbo 仅有轻微的效果损失。同时发布 [ComfyUI 节点](https://registry.comfy.org/nodes/bluepixel-comfyui)，支持 BluePixel-Turbo 和 BluePixel-Turbo-INT8 两个模型。感谢魔搭社区在 ModelScope 上提供模型下载：[BluePixel-Turbo](https://www.modelscope.cn/models/vivo/BluePixel-Turbo) 和 [BluePixel-Turbo-INT8](https://www.modelscope.cn/models/vivo/BluePixel-Turbo-INT8)。

2026.11.03 — 我们将 BluePixel 更新为 **SATP 蒸馏版本**，通过教师–学生蒸馏，以显著更少的采样步数和更低的推理成本实现高质量图像编辑。本次更新为尚未进行额外后训练的中间检查点。后续计划开展包括强化学习在内的进一步优化，以提升美学质量与指令遵循能力。🤗 [Hugging Face](https://huggingface.co/openvivo/BluePixel-Turbo/)

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

### ComfyUI 工作流

安装 [BluePixel ComfyUI 节点](https://registry.comfy.org/nodes/bluepixel-comfyui)，然后下载对应工作流并拖入 ComfyUI：

- [BluePixel-Turbo 4K 编辑工作流](workflows/bluepixel_edit_4k.json)
- [BluePixel-Turbo-INT8 4K 编辑工作流](workflows/bluepixel_edit_4k_int8.json)

按照节点说明准备模型，在加载节点中选择模型，并在 `Load Image` 节点中选择输入图片。INT8 工作流还需要 BluePixel-Turbo 的相应组件。

### 环境要求

```bash
pip install -r requirements.txt
```

运行示例前，下载完整模型：

```bash
hf download openvivo/BluePixel-Turbo --local-dir ./BluePixel-Turbo
```

使用支持 BF16 的 CUDA 环境。依赖要求包括 `diffusers>=0.40.0`，完整列表见 `requirements.txt`。

### 单图编辑

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

### INT8 模型

INT8 权重版本与 FP32 使用同一个推理脚本：当 `transformer/config.json` 中带有
`quantization_config` 时自动切换为 INT8 加载（weight-only int8，per-output-channel
scale），否则按 FP32 加载。INT8 目录只包含 transformer，其余组件（vae /
text_encoder / tokenizer / scheduler）需通过 `--base-model-dir` 指向 FP32 发布目录。

```bash
CUDA_VISIBLE_DEVICES=0 python vins_infer.py \
    --model-dir /path/to/bluepixel_..._int8 \
    --base-model-dir /path/to/BluePixel-Turbo \
    --jsonl /path/to/your_tasks.jsonl \
    --out-dir ./output_int8 \
    --bucket-area 16777216 \
    --int8-compute-dtype fp32
```

### 批量推理

准备一个 jsonl 文件，每行一条任务，包含 `image_name`（输入图像路径）和 `edit_prompt`（编辑指令）字段：

```json
{"image_name": "/path/to/input.png", "edit_prompt": "make the hair red"}
```

直接运行：

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

也可以通过启动脚本运行同一任务；脚本的模型路径默认为占位路径，需要设置 `MODEL_DIR`：

```bash
MODEL_DIR=BluePixel-Turbo JSONL=/path/to/your_tasks.jsonl OUT_DIR=./output \
    CUDA_VISIBLE_DEVICES=0 bash run_infer.sh
```

输出按 `--out-dir/seed{N}/` 分目录，并写出 `results.jsonl`（原始 jsonl 行 + `edit_image` 字段）。
`--resume`（默认开启）会跳过已存在的输出文件，`--no-resume` 重新生成并覆盖。

### 推荐分辨率

BluePixel 原生支持 4K，通过 `--bucket-area` 控制输出像素数。实现上与参考发布流程一致：
先按面积和宽高比选择对齐 bucket（边长按 `--bucket-multiple 64` 对齐），再做 resize+center-crop。

```python
bucket_area = {
    "1K": 1024 * 1024,      # 1048576
    "2K": 1440 * 1440,      # 2073600
    "4K": 4096 * 4096,      # 16777216（默认）
}
```

### 默认参数

| 参数 | 默认值 | 说明 |
|------|------|------|
| `--steps` | 4 | 去噪步数 |
| `--denoising-step-list` | `1000 750 500 250` | 自定义去噪时刻表 |
| `--cps-eta` | 0.9 | CPS 采样随机性（0 = 确定性 Euler） |
| `--bucket-area` | 16777216 | 目标像素数（4K = 4096×4096） |
| `--mu-mode` | `training` | sigma shift 模式（`none` / `empirical` / `training`） |
| `--dtype` | bf16 | text encoder 与 latent dtype |
| `--int8-compute-dtype` | fp32 | INT8 模型前向计算 dtype（`fp32` / `bf16`） |

### 显存优化

对于显存受限的设备，可以启用 VAE tiling：命令行添加 `--vae-tiling`。

FP32 transformer 常驻显存约 36 GB（权重 36.3 GB + VAE 2.7 GB + text encoder bf16 8.2 GB），
4K 推理建议 80 GB 以上显存；INT8 transformer 权重约 9.1 GB，显存占用显著更低。

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-dir` | str | **必填** | 模型权重目录（INT8 自动识别） |
| `--base-model-dir` | str | `None` | 存放 vae/text_encoder/tokenizer/scheduler 的目录（默认同 `--model-dir`） |
| `--jsonl` | str | **必填** | 任务列表 jsonl 文件 |
| `--out-dir` | str | **必填** | 输出目录（按 seed 分子目录） |
| `--device` | str | `cuda` | 推理设备 |
| `--image-field` | str | `image_name` | 输入图像路径字段名 |
| `--instruction-field` | str | `edit_prompt` | 编辑指令字段名 |
| `--task-types` | list | `None` | 仅处理指定 task_type |
| `--start-idx` | int | `0` | 数据集起始下标 |
| `--num-samples` | int | `None` | 处理样本数 |
| `--limit` | int | `None` | 仅处理前 N 条 |
| `--seed` | int | `42` | 随机种子 |
| `--seeds` | str | `None` | 逗号分隔的多个随机种子 |
| `--steps` | int | `4` | 去噪步数 |
| `--denoising-step-list` | list | `1000 750 500 250` | 去噪时刻表 |
| `--cps-eta` | float | `0.9` | CPS 采样随机性 |
| `--mu-mode` | str | `training` | sigma shift 模式（`none` / `empirical` / `training`） |
| `--sigma-image-seq-len` | int | `None` | 覆盖 sigma shift 使用的序列长度 |
| `--dtype` | str | `bf16` | text encoder / latent dtype |
| `--int8-compute-dtype` | str | `fp32` | INT8 前向计算 dtype |
| `--bucket-ratios` | list | 默认 14 种 | bucket 宽高比集合 |
| `--bucket-area` | int | `16777216` | bucket 目标面积（4K） |
| `--bucket-multiple` | int | `64` | bucket 边长对齐粒度 |
| `--vae-tiling` | flag | `False` | 启用 VAE tiling 以节省显存 |
| `--no-resume` | flag | `False` | 重跑全部选中任务并覆盖输出 |
| `--no-results-jsonl` | flag | `False` | 不写出 results.jsonl |
| `--dry-run` | flag | `False` | 只打印任务计划，不执行推理 |

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

## 项目结构

```
BluePixelModel/
├── vins_infer.py                     # 单卡推理脚本（FP32 / INT8 自动识别，支持断点续传）
├── run_infer.sh                      # 单卡推理启动脚本（一键测试）
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
