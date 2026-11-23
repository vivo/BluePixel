"""Unit tests for the released-inference numerics in vins_infer.py.

These lock down the pieces that must match the reference release pipeline:
resolution buckets, sigma schedule, CPS step, and INT8 detection/loading.
"""
from __future__ import annotations

import math

import pytest
import torch

import vins_infer as vi


class _SchedulerConfig:
    num_train_timesteps = 1000
    use_dynamic_shifting = True
    time_shift_type = "exponential"
    shift = 3.0
    shift_terminal = None


def test_bucket_set_matches_reference_construction():
    buckets = vi.ResolutionBucketSet(target_area=16777216, multiple=64)
    assert len(buckets) == len(vi.DEFAULT_BUCKET_RATIOS)
    square = buckets.select(1000, 1000)
    assert square["width"] == square["height"] == 4096
    assert square["width"] % 64 == 0
    for bucket in buckets:
        assert bucket["width"] % 64 == 0
        assert bucket["height"] % 64 == 0
        assert bucket["width"] * bucket["height"] <= 16777216


def test_bucket_selects_by_closest_log_aspect_ratio():
    buckets = vi.ResolutionBucketSet(target_area=1048576, multiple=64)
    # 3:1 is closest to 7:3 (0.251) rather than 4:1 (0.288) in log space.
    wide = buckets.select(1920, 640)
    assert wide["ratio"] == "7:3"
    tall = buckets.select(640, 1920)
    assert tall["ratio"] == "1:4"


def test_training_mu_is_resolution_only_and_capped():
    assert vi.compute_training_mu(4096) == pytest.approx(0.00016927 * 4096 + 0.45666666)
    assert vi.compute_training_mu(65536) == vi.compute_training_mu(4096)


def test_sigmas_for_reference_schedule():
    sigmas = vi.compute_sigmas(
        [1000, 750, 500, 250],
        _SchedulerConfig(),
        image_seq_len=4096,
        num_steps=4,
        mu_mode="training",
    )
    assert len(sigmas) == 4
    mu = vi.compute_training_mu(4096)
    expected_first = math.exp(mu) / (math.exp(mu) + (1 / 1.0 - 1) ** 1.0)
    assert sigmas[0] == pytest.approx(expected_first)
    assert all(b < a for a, b in zip(sigmas, sigmas[1:]))


def test_none_mu_mode_uses_static_shift():
    sigmas = vi.compute_sigmas(
        [1000, 500],
        _SchedulerConfig(),
        image_seq_len=4096,
        num_steps=2,
        mu_mode="none",
    )
    shift = 3.0
    expected = shift * 1.0 / (1 + (shift - 1) * 1.0)
    assert sigmas[0] == pytest.approx(expected)


def test_cps_step_matches_euler_when_eta_zero():
    scheduler = vi.CPSScheduler()
    scheduler.set_timesteps(sigmas=[0.75, 0.5])
    scheduler._step_index = 0
    sample = torch.randn(1, 4, 8, 8)
    velocity = torch.randn(1, 4, 8, 8)

    out = scheduler.step(velocity, scheduler.timesteps[0], sample, return_dict=False)[0]
    expected = sample + (torch.tensor(0.5) - torch.tensor(0.75)) * velocity
    assert torch.allclose(out.float(), expected.float(), atol=1e-6)


def test_cps_step_is_deterministic_reproducible_with_eta():
    def run():
        scheduler = vi.CPSScheduler()
        scheduler.set_eta(0.9)
        scheduler.set_timesteps(sigmas=[0.75, 0.5])
        scheduler._step_index = 0
        g = torch.Generator().manual_seed(7)
        return scheduler.step(
            torch.zeros(1, 4, 4, 4), scheduler.timesteps[0], torch.zeros(1, 4, 4, 4),
            generator=g, return_dict=False,
        )[0]

    assert torch.equal(run(), run())


def test_cps_step_has_no_noise_at_terminal_sigma():
    scheduler = vi.CPSScheduler()
    scheduler.set_eta(1.0)
    scheduler.set_timesteps(sigmas=[0.25, 0.0])
    scheduler._step_index = 0
    sample = torch.randn(1, 4, 4, 4)
    velocity = torch.randn(1, 4, 4, 4)
    out = scheduler.step(velocity, scheduler.timesteps[0], sample, return_dict=False)[0]
    expected = sample + (torch.tensor(0.0) - torch.tensor(0.25)) * velocity
    assert torch.allclose(out.float(), expected.float(), atol=1e-6)


def test_latent_ids_layout():
    latents = torch.zeros(1, 128, 3, 5)
    ids = vi._prepare_latent_ids(latents)
    assert ids.shape == (1, 15, 4)
    assert ids[0, :, 0].unique().tolist() == [0]
    assert ids[0, :, 3].unique().tolist() == [0]


def test_ref_image_ids_start_at_scale():
    latents = torch.zeros(1, 128, 2, 2)
    ids = vi._build_ref_image_ids_dcae(latents, t_scale=10, t_index=0)
    assert ids.shape == (1, 4, 4)
    assert ids[..., 0].unique().tolist() == [10]


def test_pack_unpack_roundtrip():
    latents = torch.randn(2, 128, 4, 6)
    tokens = vi.pack_flux2_latents(latents)
    assert tokens.shape == (2, 24, 128)
    back = vi.unpack_flux2_latents(tokens, height=4, width=6)
    assert torch.equal(back, latents)


def test_unpack_rejects_wrong_token_count():
    with pytest.raises(ValueError, match="Token count mismatch"):
        vi.unpack_flux2_latents(torch.zeros(1, 7, 128), height=4, width=6)


def test_preprocess_ref_image_is_normalized_and_sized():
    from PIL import Image

    image = Image.new("RGB", (100, 50), color=(255, 0, 0))
    tensor = vi.preprocess_ref_image_pil(image, target_height=64, target_width=64)
    assert tuple(tensor.shape) == (3, 64, 64)
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0
    assert float(tensor[0].min()) == pytest.approx(1.0)


def test_int8_linear_dequantizes_and_applies_bias():
    linear = vi.Int8Linear(4, 3, has_bias=True)
    weight = torch.tensor(
        [[1, -2, 3, -4], [0, 1, 0, -1], [2, 2, -2, -2]], dtype=torch.int8
    )
    linear.weight.copy_(weight)
    linear.weight_scale.copy_(torch.tensor([0.5, 2.0, 1.0]))
    linear.bias.data.copy_(torch.tensor([1.0, -1.0, 0.0]))

    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = linear(x)
    expected = torch.nn.functional.linear(x, weight.float()) * torch.tensor([0.5, 2.0, 1.0])
    expected = expected + torch.tensor([1.0, -1.0, 0.0])
    assert torch.allclose(out, expected)


def test_read_quantization_config_returns_none_for_fp32_dir(tmp_path):
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text('{"in_channels": 128}', encoding="utf-8")
    assert vi.read_quantization_config(str(transformer_dir)) is None


def test_read_quantization_config_detects_int8(tmp_path):
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        '{"quantization_config": {"quant_method": "int8_weight_only", '
        '"quantized_keys": ["a.weight"], "scale_suffix": "_scale"}}',
        encoding="utf-8",
    )
    cfg = vi.read_quantization_config(str(transformer_dir))
    assert cfg["quant_method"] == "int8_weight_only"
    assert cfg["quantized_keys"] == ["a.weight"]


def test_load_pipeline_rejects_bad_int8_compute_dtype(tmp_path):
    with pytest.raises(ValueError, match="int8_compute_dtype"):
        vi.load_pipeline(str(tmp_path), int8_compute_dtype="int4")


def test_load_pipeline_requires_transformer_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="transformer/"):
        vi.load_pipeline(str(tmp_path))
