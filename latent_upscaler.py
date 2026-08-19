"""Small, self-contained MiniMax H3 latent upscaler.

The network and normalization constants mirror
LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.  It is intentionally kept
inside Auto Director so the dual-sampling node does not require a second
custom-node package.  Only the video tensor is passed through the network;
the H3 audio tensor remains untouched by the caller.
"""

from __future__ import annotations

import glob
import math
import os
import re

import folder_paths
import torch
import torch.nn as nn
import torch.nn.functional as F


FOLDER = "latent_upscale_models"
if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))

MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293249435425,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


def available_models():
    paths = []
    root = folder_paths.get_folder_paths(FOLDER)[0]
    for ext in ("*.pth", "*.safetensors"):
        paths.extend(glob.glob(os.path.join(root, ext)))
    names = sorted(os.path.basename(path) for path in paths)
    return names or [f"(请将 H3 latent 放大模型放入: {root})"]


def _norm(device, dtype):
    return (torch.tensor(MEAN, device=device, dtype=dtype).view(1, 24, 1, 1, 1),
            torch.tensor(STD, device=device, dtype=dtype).view(1, 24, 1, 1, 1))


def _norm_state_dict(state):
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if any(key.startswith("upscaler.") for key in state):
        state = {key[9:]: value for key, value in state.items() if key.startswith("upscaler.")}
    return {
        key: value.to(torch.float16) if torch.is_tensor(value) and value.dtype == torch.float8_e4m3fn else value
        for key, value in state.items()
    }


def _load_state(path):
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        return _norm_state_dict(load_file(path, device="cpu"))
    return _norm_state_dict(torch.load(path, map_location="cpu", weights_only=False))


def _group_norm(channels):
    return nn.GroupNorm(32, channels)


def _zero(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class _Temporal(nn.Module):
    def __init__(self, channels, kernel=5):
        super().__init__()
        self.norm = _group_norm(channels)
        self.dwconv = nn.Conv3d(channels, channels, (kernel, 1, 1), padding=(kernel // 2, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, value):
        identity = value
        value = self.dwconv(F.silu(self.norm(value)))
        return identity + self.pwconv(value)


class _Block(nn.Module):
    def __init__(self, channels, emb_channels=64):
        super().__init__()
        self.in_layers = nn.Sequential(_group_norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, channels * 2))
        self.out_norm = _group_norm(channels)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(0.1), _zero(nn.Conv3d(channels, channels, 3, padding=1)))

    def forward(self, value, emb):
        hidden = self.in_layers(value)
        scale, shift = torch.chunk(self.emb_layers(emb).to(hidden.dtype), 2, dim=1)
        hidden = self.out_norm(hidden) * (1 + scale[..., None, None, None]) + shift[..., None, None, None]
        return value + self.out_layers(hidden)


class _Resizer(nn.Module):
    def __init__(self, channels=512, in_blocks=12, out_blocks=12, temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(24, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.in_blocks = nn.ModuleList()
        for index in range(in_blocks):
            self.in_blocks.append(_Block(channels))
            if temporal_every and index % temporal_every == 0:
                self.in_blocks.append(_Temporal(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for index in range(out_blocks):
            self.out_blocks.append(_Block(channels))
            if temporal_every and index % temporal_every == 0:
                self.out_blocks.append(_Temporal(channels, temporal_kernel))
        self.norm_out = _group_norm(channels)
        self.conv_out = nn.Conv3d(channels, 24, 3, padding=1)

    def forward(self, value, target_size, scale):
        emb = self.embed(torch.tensor([[float(scale - 1)]], dtype=value.dtype, device=value.device))
        value = self.conv_in(value)
        for block in self.in_blocks:
            value = block(value, emb) if isinstance(block, _Block) else block(value)
        value = F.interpolate(value, size=target_size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            value = block(value, emb) if isinstance(block, _Block) else block(value)
        return self.conv_out(F.silu(self.norm_out(value)))


_CACHE = {}


def _arch(state):
    cfg = {"channels": int(state.get("conv_in.weight", torch.empty(512, 24, 3, 3, 3)).shape[0]), "in_blocks": 12, "out_blocks": 12, "temporal_every": 2, "temporal_kernel": 5}
    in_ids, out_ids = set(), set()
    has_temporal = False
    for key, value in state.items():
        match = re.match(r"in_blocks\.(\d+)\.in_layers", key)
        if match:
            in_ids.add(int(match.group(1)))
        match = re.match(r"out_blocks\.(\d+)\.in_layers", key)
        if match:
            out_ids.add(int(match.group(1)))
        if key.endswith("dwconv.weight"):
            has_temporal = True
            cfg["temporal_kernel"] = int(value.shape[2])
    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)
    if not has_temporal:
        cfg["temporal_every"] = 0
    return cfg


def _get_model(name, device, precision):
    key = (str(name), str(device), str(precision))
    if key in _CACHE:
        return _CACHE[key]
    if not name or str(name).startswith("("):
        raise ValueError("未选择有效的 H3 latent 放大模型")
    root = folder_paths.get_folder_paths(FOLDER)[0]
    path = os.path.join(root, str(name))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"H3 latent 放大模型不存在: {path}")
    state = _load_state(path)
    cfg = _arch(state)
    network = _Resizer(**cfg)
    network.load_state_dict(state, strict=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(str(precision), torch.float32)
    network = network.to(device=device, dtype=dtype).eval()
    _CACHE[key] = network
    return network


def upscale_video(samples, model_name, target_height, target_width, device="cuda", precision="fp32"):
    """Return a CPU tensor with shape [B,24,T,H,W], audio untouched by design."""
    if not torch.is_tensor(samples) or samples.ndim != 5 or int(samples.shape[1]) != 24:
        raise ValueError(f"H3 latent 放大需要 [B,24,T,H,W] 视频 latent，实际为 {tuple(getattr(samples, 'shape', ())) }")
    source_h, source_w = int(samples.shape[-2]), int(samples.shape[-1])
    target_h, target_w = int(target_height), int(target_width)
    if (source_h, source_w) == (target_h, target_w):
        return samples
    run_device = torch.device(device if str(device) == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(str(precision), torch.float32)
    model = _get_model(model_name, run_device, precision)
    source_dtype = samples.dtype
    with torch.no_grad():
        value = samples.to(run_device, dtype=dtype)
        mean, std = _norm(run_device, dtype)
        value = (value - mean) / std
        value = model(value, (int(value.shape[2]), target_h, target_w), math.sqrt((target_h * target_w) / max(1, source_h * source_w)))
        value = value * std + mean
    return value.to(device="cpu", dtype=source_dtype).contiguous()
