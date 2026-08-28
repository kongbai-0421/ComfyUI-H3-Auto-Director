"""Project runner primitives for MiniMax H3 in ComfyUI.

The controller queues the next copy of the current workflow after a segment
has been saved. It deliberately uses numbered project slots, so a rerun never
silently consumes the newest rejected cache.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib
import inspect
from collections import OrderedDict
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import wave
from collections import deque
from pathlib import Path

import folder_paths
import nodes
import torch
import node_helpers
import comfy.model_management as model_management
import comfy.memory_management
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
try:
    # NestedTensor is native to current H3 AV sampling.  It is unavailable in
    # older ComfyUI builds, which use the legacy list-of-streams contract.
    import comfy.nested_tensor as _h3_nested_tensor
except (ImportError, AttributeError):
    _h3_nested_tensor = None
from .sampling_switch import LEGACY_MODE, NATIVE_MODE, apply_h3_sampling, ensure_h3_layout_refresh

try:
    from comfy_extras import nodes_minimax_h3 as _minimax_h3
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo as _H3ReferenceToVideo
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide as _H3AddGuide
except ImportError:
    _minimax_h3 = None
    _H3ReferenceToVideo = None
    _H3AddGuide = None

try:
    from . import latent_upscaler as _h3_latent_upscaler
except ImportError:
    _h3_latent_upscaler = None


def _encode_ref_audio(audio_vae, audio):
    """Encode a soundtrack with the H3 audio VAE.

    ``_encode_ref_audio`` is a module-level function in current ComfyUI
    (comfy_extras.nodes_minimax_h3).  Older builds exposed it on the
    MiniMaxH3ReferenceToVideo class, so fall back to that when absent.
    """
    fn = getattr(_minimax_h3, "_encode_ref_audio", None)
    if fn is None:
        fn = getattr(_H3ReferenceToVideo, "_encode_ref_audio", None)
    if fn is None:
        raise RuntimeError("当前 ComfyUI 未提供 _encode_ref_audio 音频编码函数")
    return fn(audio_vae, audio)

try:
    import av
except ImportError:
    av = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

try:
    from safetensors.torch import load_file as st_load, save_file as st_save
except ImportError:
    st_load = st_save = None

try:
    from safetensors import safe_open as st_safe_open
except ImportError:
    st_safe_open = None

LOG = logging.getLogger("h3_auto_director")
FPS = 24.0
# Diagnostic mode: keep the complete generated timeline while investigating
# continuation joins. Set the environment variable to ``1`` to restore the
# old automatic context-prefix crop after the comparison is complete.
AUTO_CONTEXT_CROP_DEFAULT = False
FRAME_CONTEXT_DEFAULT = 22
PROMPT_CACHE_MAX_PROJECTS = 2
PROMPT_DISK_CACHE_SCHEMA = 1
_PROMPT_CONDITIONING_CACHE = OrderedDict()
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCE_TOTAL = 12
PROJECT_ROOT_NAME = "h3_project"

VIDEO_FORMATS = {"mp4": "mp4", "mkv": "matroska", "webm": "webm", "mov": "mov"}
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".avi")
VIDEO_CODECS = {
    "h264": {"cpu": "libx264", "gpu": ("h264_nvenc", "h264_qsv", "h264_amf")},
    "hevc": {"cpu": "libx265", "gpu": ("hevc_nvenc", "hevc_qsv", "hevc_amf")},
    "vp9": {"cpu": "libvpx-vp9", "gpu": ()},
    "av1": {"cpu": "libaom-av1", "gpu": ("av1_nvenc", "av1_qsv", "av1_amf")},
}
QUALITY_CHOICES = ("最高质量", "高质量", "平衡", "快速")


def _direct_video_file(value):
    """Resolve a directly selected/uploaded video file.

    The standalone video tool intentionally accepts a file, never a
    directory.  Browser uploads return an ``input/``-relative path while
    native ComfyUI workflows may provide an absolute Windows path, so both
    forms are normalized here.  Keeping this check in one helper prevents
    metadata probing and decoding from disagreeing about the input.
    """
    raw = str(value or "").strip().strip('"')
    if not raw:
        raise ValueError("请直接上传或填写视频文件（支持 MP4、MKV、WebM、MOV、AVI），不能填写目录")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        normalized = raw.replace("\\", "/")
        if normalized.startswith("input/"):
            normalized = normalized[6:]
        input_candidate = Path(folder_paths.get_input_directory()) / normalized
        candidate = input_candidate if input_candidate.exists() else candidate
    candidate = candidate.resolve()
    if candidate.is_dir():
        raise ValueError("视频加载节点仅支持直接加载视频文件，不能加载目录：%s" % candidate)
    if not candidate.is_file():
        raise ValueError("视频文件不存在：%s" % candidate)
    if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("不支持的视频格式：%s（支持：%s）" % (candidate.suffix or "无扩展名", ", ".join(VIDEO_EXTENSIONS)))
    return candidate
ENCODER_DEVICES = ("CPU", "GPU")
COLOR_CORRECTION_CHOICES = ("关闭", "匹配首段", "匹配上段")
CONTEXT_DIR_NAME = "context"
CONTEXT_STAGE1_DIR_NAME = "context_stage1"
DUAL_UPSCALE_CHOICES = ("普通插值", "H3 Latent 学习型放大", "普通放大模型", "RTX Video Super Resolution", "自动（RTX→普通模型→插值）")
_MOTION_CONTEXT_MARKER = "_h3_auto_director_motion_context"
_NATIVE_CONTEXT_KEY = "_h3_auto_director_native_context"
_LAST_STAGE1_CONTEXT = None
_LAST_MOTION_CONTEXT_TRIM = None


def _h3_canvas_dimensions(width, height):
    """Return the exact spatial grid consumed by MiniMax H3.

    H3 packs video latents at a 16x VAE stride but requires the pixel canvas
    itself to be a multiple of ``CANVAS_MULTIPLE`` (32 in the official node).
    Normalize at every public boundary so manually entered or legacy workflow
    values cannot be silently truncated by ``_empty_av_latent``.
    """
    multiple = int(getattr(_minimax_h3, "CANVAS_MULTIPLE", 32) or 32)
    multiple = max(32, multiple)
    try:
        w = int(round(float(width) / multiple) * multiple)
        h = int(round(float(height) / multiple) * multiple)
    except (TypeError, ValueError):
        raise ValueError("H3 分辨率必须是有效的宽度和高度")
    return max(multiple, w), max(multiple, h)


def _output_root():
    return Path(folder_paths.get_output_directory()).resolve()


def _find_ffmpeg():
    """Find an ffmpeg executable even when ComfyUI was launched without it on PATH."""
    candidates = []
    configured = os.environ.get("FFMPEG_PATH", "").strip().strip('"')
    if configured:
        candidates.append(configured)
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(found)
    if imageio_ffmpeg is not None:
        try:
            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            pass
    comfy_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        comfy_root.parent / "python" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64.exe",
        comfy_root.parent / "python_embeded" / "ffmpeg.exe",
        comfy_root / "ffmpeg.exe",
    ])
    for value in candidates:
        path = Path(value)
        if path.is_file():
            return str(path)
    return None


def _release_video_memory():
    """Release temporary Python/CUDA allocations between video chunks."""
    gc.collect()
    try:
        model_management.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _safe_project_dir(project_id: str, root: str = "h3_projects") -> Path:
    project_id = str(project_id).strip()
    if not project_id or project_id in {".", ".."} or any(c in project_id for c in "\\/:*"):
        raise ValueError("Invalid H3 Auto Director project id")
    root = str(root or "").strip().strip("/\\")
    if any(c in root for c in ":*?<>|\n\r") or root in {".", ".."}:
        raise ValueError("Invalid H3 Auto Director project folder")
    # New layout: output/h3_project/<folder>. A custom Plan output_root is
    # the folder name; the old default keeps project_id as that folder.
    folder = project_id if root in {"", ".", "h3_projects"} else root
    base = (_output_root() / PROJECT_ROOT_NAME / folder).resolve()
    if _output_root() not in base.parents:
        raise ValueError("Project directory must stay inside ComfyUI output")
    return base


def _legacy_project_dirs(project_id: str, root: str):
    """Candidates used only to continue projects created by older layouts."""
    root = str(root or "").strip().strip("/\\")
    candidates = []
    if root not in {"", ".", "h3_projects"}:
        candidates.append(_output_root() / root / project_id)
    candidates.extend([_output_root() / project_id, _output_root() / "h3_projects" / project_id])
    return [path.resolve() for path in candidates if _output_root() in path.resolve().parents]


def _find_legacy_project(project_id: str, root: str, current: Path | None = None):
    current = current.resolve() if current is not None else None
    for path in _legacy_project_dirs(project_id, root):
        if current is not None and path == current:
            continue
        if (path / "json" / "project.json").exists() or (path / "project.json").exists():
            return path
    return None


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_digest(value):
    """Create a deterministic digest for prompt-cache manifest entries."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_model_identity(*objects):
    """Return a conservative identity for the loaded encoder/VAE classes.

    ComfyUI does not expose the source checkpoint path consistently across
    releases.  The cache schema and class/config identity still prevent a
    cache written by another encoder family or core implementation from being
    silently reused.  Reference file fingerprints below catch changed assets.
    """
    result = []
    for value in objects:
        if value is None:
            result.append(None)
            continue
        patcher = getattr(value, "patcher", None)
        model = getattr(patcher, "model", None) if patcher is not None else None
        config = getattr(model, "model_config", None)
        result.append({
            "object": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "patcher": f"{patcher.__class__.__module__}.{patcher.__class__.__qualname__}" if patcher else None,
            "model": f"{model.__class__.__module__}.{model.__class__.__qualname__}" if model else None,
            "config": f"{config.__class__.__module__}.{config.__class__.__qualname__}" if config else None,
            "latent_channels": getattr(value, "latent_channels", None),
            "audio_sample_rate": getattr(value, "audio_sample_rate", None),
        })
    return result


def _reference_file_marker(plan, ref):
    """Fingerprint a local reference or a generated previous-segment clip."""
    if not isinstance(ref, dict):
        return {"value": str(ref)}
    kind = str(ref.get("type", "image")).lower()
    value = ref.get("path") or ref.get("name")
    if kind == "previous_segment_video":
        try:
            path, _ = _paths(plan, int(ref.get("segment_index", 0)), for_context=True)
        except Exception:
            path = None
    elif value:
        try:
            path = (_input_root() / _reference_name(value)).resolve()
        except Exception:
            path = None
    else:
        path = None
    marker = {key: ref.get(key) for key in (
        "type", "path", "name", "segment_index", "video_number", "video_audio_enabled",
        "start_frame", "source_frames", "reference_frames") if key in ref}
    if path is not None:
        marker["resolved_path"] = str(path)
        try:
            stat = path.stat()
            marker["size"] = int(stat.st_size)
            marker["mtime_ns"] = int(stat.st_mtime_ns)
        except OSError:
            marker["missing"] = True
    return marker


def _prompt_disk_fingerprint(plan, generation_index, width, height, ref_image_size,
                             context_length, ref_short_edge, model_identity):
    refs = _cache_segment_references(plan, generation_index)
    seg = _segment(plan, generation_index)
    prompt = _previous_video_prompt(seg.get("prompt", ""), refs)
    # Reference conditioning does not depend on the generation canvas when
    # using a fixed short-edge policy (manual) or H3's max policy.  The match
    # policy scales images to the generation pixel area, so width/height must
    # remain part of its cache identity.
    ref_mode = str(ref_image_size or "match").lower()
    width, height = _h3_canvas_dimensions(width, height)
    resolution = ({"width": int(width), "height": int(height)}
                  if ref_mode not in {"manual", "max"} else None)
    value = {
        "schema": PROMPT_DISK_CACHE_SCHEMA,
        "mode": plan.get("mode", "director"),
        "project_id": plan.get("project_id"),
        "global_reference_set": bool(plan.get("global_reference_set", True)),
        "global_assets": plan.get("global_assets", []),
        "continuation_mode": plan.get("continuation_mode", True),
        "video_continuation": plan.get("video_continuation"),
        "segment_index": int(generation_index),
        "segment": seg,
        "prompt": prompt,
        "references": [_reference_file_marker(plan, ref) for ref in refs],
        "resolution": resolution,
        "ref_image_size": str(ref_image_size),
        "context_length": int(context_length),
        "ref_short_edge": _nearest_multiple(ref_short_edge),
        "models": model_identity,
        "core": getattr(_minimax_h3, "__file__", None),
    }
    return _stable_digest(value), value


def _prompt_disk_paths(plan):
    root = Path(plan["project_dir"]) / "cache" / "prompt_embeddings"
    return root, root / "manifest.json"


def _cpu_copy(value):
    """Recursively detach conditioning tensors so torch.save never retains CUDA."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _allow_prompt_cache_safe_globals():
    """Allow ComfyUI's AV container when loading trusted prompt caches.

    PyTorch 2.6 changed ``torch.load``'s default safe-unpickler policy. H3
    conditioning can legitimately contain ComfyUI's ``NestedTensor`` AV
    container, so a weights-only load otherwise rejects a valid cache file.
    Older PyTorch releases do not expose ``add_safe_globals`` and simply skip
    this compatibility registration.
    """
    add_safe_globals = getattr(getattr(torch, "serialization", None),
                               "add_safe_globals", None)
    if add_safe_globals is None:
        return
    try:
        nested_module = importlib.import_module("comfy.nested_tensor")
        nested_tensor = getattr(nested_module, "NestedTensor", None)
        if nested_tensor is not None:
            add_safe_globals([nested_tensor])
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        LOG.debug("H3 Auto Director: 无法注册 NestedTensor 安全缓存类型：%s", exc)


def _load_torch_cache(path):
    _allow_prompt_cache_safe_globals()
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _load_prompt_disk_manifest(plan):
    _, manifest_path = _prompt_disk_paths(plan)
    try:
        value = _load_json(manifest_path, {})
    except (OSError, ValueError, json.JSONDecodeError):
        LOG.warning("H3 Auto Director: 提示词向量磁盘缓存清单损坏，将重新编码")
        return {}
    return value if isinstance(value, dict) and value.get("schema") == PROMPT_DISK_CACHE_SCHEMA else {}


def _save_prompt_disk_entry(plan, manifest, generation_index, fingerprint, conditioning, details):
    root, manifest_path = _prompt_disk_paths(plan)
    root.mkdir(parents=True, exist_ok=True)
    filename = "segment_%05d.pt" % int(generation_index)
    path = root / filename
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(_cpu_copy(conditioning), str(tmp))
    os.replace(tmp, path)
    manifest.setdefault("segments", {})[str(int(generation_index))] = {
        "fingerprint": fingerprint, "file": filename, "details": details,
    }
    _atomic_json(manifest_path, manifest)


def _bool_setting(value, default=False):
    """Normalize ComfyUI/JSON boolean values, including legacy strings."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"", "0", "false", "off", "no", "关闭", "否"}:
        return False
    if text in {"1", "true", "on", "yes", "开启", "是"}:
        return True
    return bool(default)


def _prompt_cache_all_enabled(plan):
    return _bool_setting(plan.get("cache_prompt_embeddings", False), False)


def _disk_cache_enabled(plan):
    return _bool_setting(plan.get("cache_prompt_embeddings_to_disk", False), False)


def _prompt_disk_global_details(plan, width, height, ref_image_size, context_length,
                                ref_short_edge, model_identity, use_manual_ref_short_edge=False):
    """Build the manifest-wide signature used by both eager and delayed cache writes."""
    effective_mode = str("manual" if use_manual_ref_short_edge else ref_image_size or "match").lower()
    width, height = _h3_canvas_dimensions(width, height)
    resolution = ({"width": int(width), "height": int(height)}
                  if effective_mode not in {"manual", "max"} else None)
    return {
        "schema": PROMPT_DISK_CACHE_SCHEMA,
        "mode": plan.get("mode", "director"),
        "project_id": plan.get("project_id"),
        "global_reference_set": bool(plan.get("global_reference_set", True)),
        "global_assets": plan.get("global_assets", []),
        "resolution": resolution,
        "ref_image_size": effective_mode,
        "context_length": int(context_length),
        "ref_short_edge": _nearest_multiple(ref_short_edge),
        "models": model_identity,
    }


def _align_frames(frames: int) -> int:
    frames = max(5, int(frames))
    while frames % 17 != 5:
        frames += 1
    return frames


def _align_frames_nearest(frames: int) -> int:
    """Choose the nearest valid H3 duration for a context-prefixed timeline."""
    target = max(5, int(frames))
    lower = max(5, target - ((target - 5) % 17))
    upper = lower + 17
    return lower if target - lower <= upper - target else upper


def _segment(plan, index: int):
    segs = plan.get("segments", [])
    if index < 1 or index > len(segs):
        raise ValueError("segment_index %d is outside the project" % index)
    value = dict(segs[index - 1])
    value.setdefault("prompt", "")
    value.setdefault("duration", plan.get("duration", 5.0))
    value.setdefault("audio_restart", False)
    value.setdefault("continue_video", index > 1)
    return value


def _parse_segment_numbers(value, label):
    """Parse a 1-based comma-separated segment list entered by the user."""
    text = str(value or "").strip().replace("，", ",")
    if not text:
        return set()
    result = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            number = int(part)
        except ValueError as exc:
            raise ValueError(f"{label}必须是用逗号分隔的片段编号，例如：2, 5, 8") from exc
        if number < 1:
            raise ValueError(f"{label}中的片段编号必须从 1 开始")
        result.add(number)
    return result


def _video_context_enabled(plan):
    """Retain legacy plans while allowing transfer plans to split AV policies."""
    # Newer plan writers may persist the optional split policy as null when
    # it is not explicitly configured.  ``dict.get(key, fallback)`` does not
    # use the fallback for an existing null value, so bool(None) accidentally
    # disabled video continuation for otherwise-enabled projects.  Treat null
    # as absent and inherit the legacy/global continuation switch.
    value = plan.get("video_continuation")
    if value is None:
        value = plan.get("continuation_mode", True)
    return bool(value)


def _use_previous_video_reference(plan, generation_index: int) -> bool:
    """Read the previous-video reference flag from the target segment only."""
    if int(generation_index) <= 1:
        return False
    seg = _segment(plan, int(generation_index))
    return bool(seg.get("use_previous_video_reference", False))


def _segment_reference_specs(plan, generation_index: int):
    """Return user references plus a runtime-only previous-video reference when enabled."""
    seg = _segment(plan, int(generation_index))
    global_set = bool(plan.get("global_reference_set", True))
    refs = list(plan.get("global_assets", [])) if global_set else list(seg.get("references", []))
    if global_set and int(generation_index) != 1:
        # Global mode reuses the first segment's files as ordinary references,
        # but timed insertion belongs only to the segment where it was set.
        # Strip insertion metadata on later segments so a guide at (say) 2s
        # is not injected into every segment sharing the global set.
        refs = [
            dict(ref, insert_seconds=0.0, insert_frames=0)
            if isinstance(ref, dict) else ref
            for ref in refs
        ]
    if _use_previous_video_reference(plan, generation_index):
        video_count = sum(1 for ref in refs if isinstance(ref, dict) and str(ref.get("type", "image")).lower() in {"video", "transfer_video_segment"})
        if video_count > 1:
            raise ValueError(f"第 {int(generation_index)} 段启用上片段视频参考时，最多只能上传 1 个视频参考素材")
        refs.append({
            "type": "previous_segment_video",
            "segment_index": int(generation_index) - 1,
            "video_number": video_count + 1,
            "video_audio_enabled": False,
        })
    _validate_reference_limits(refs, f"第 {int(generation_index)} 段参考素材")
    return refs


def _previous_video_prompt(prompt, refs):
    """Prefix the generated prompt with the runtime reference's tail-to-head instruction."""
    previous = next((ref for ref in refs if isinstance(ref, dict) and ref.get("type") == "previous_segment_video"), None)
    if previous is None:
        return str(prompt or "")
    number = int(previous.get("video_number") or 1)
    prefix = (
        f"<Video {number}> is the immediately preceding generated segment. "
        "Use its final tail frame as the first-frame state of this segment, "
        "then generate the following motion as a continuous visual continuation "
        "with consistent subject identity, composition, lighting, and camera direction."
    )
    text = str(prompt or "").strip()
    if text.startswith(prefix):
        return text
    return prefix + ("\n\n" + text if text else "")


def _plan_from_id(project_id: str, root: str):
    project_dir = _safe_project_dir(project_id, root)
    manifest = project_dir / "json" / "project.json"
    if not manifest.exists():
        manifest = project_dir / "project.json"
    if not manifest.exists():
        for legacy in _legacy_project_dirs(str(project_id), root):
            candidate = legacy / "json" / "project.json"
            if not candidate.exists():
                candidate = legacy / "project.json"
            if candidate.exists():
                project_dir, manifest = legacy, candidate
                break
    plan = _load_json(manifest)
    if not isinstance(plan, dict):
        raise FileNotFoundError("H3 project manifest not found: %s" % project_dir)
    plan["project_dir"] = str(project_dir)
    return plan


def _output_filename(value=""):
    name = str(value or "").strip().strip('"')
    if not name or name == "h3_projects":
        return "H3"
    name = Path(name).stem
    if not name or name in {".", ".."} or any(c in name for c in "\\/:*?<>|\n\r"):
        raise ValueError("输出文件名只能包含文件名，不能包含路径或特殊字符")
    return name


def _audio_filename(value="", index=1):
    """Sanitize a per-segment audio filename without affecting latent names."""
    text = str(value or "").strip().strip('"')
    if not text:
        return "H3_%05d.wav" % int(index)
    name = Path(text).name
    if not name or name in {".", ".."} or any(c in name for c in "\\/:*?<>|\n\r"):
        raise ValueError("音频文件名不能包含路径或特殊字符")
    if not Path(name).suffix:
        name += ".wav"
    if Path(name).suffix.lower() != ".wav":
        raise ValueError("音频文件名必须使用 .wav 扩展名")
    return name


def _audio_path(plan, index, filename="", for_write=False):
    """Return the TTS segment path, with read compatibility for old projects.

    New projects keep generated segments below ``audio/segments`` so the
    project root remains unambiguous.  Existing projects used ``audio``
    directly; they remain resumable and concatenatable without migration.
    """
    name = _audio_filename(filename, index)
    base = Path(plan["project_dir"])
    modern = base / "audio" / "segments" / name
    legacy = base / "audio" / name
    return modern if for_write or modern.exists() or not legacy.exists() else legacy


def _indexed_file(directory: Path, index: int, suffix, output_name=""):
    suffixes = (suffix,) if isinstance(suffix, str) else tuple(suffix)
    stem = _output_filename(output_name)
    preferred = directory / ("%s_%05d%s" % (stem, index, suffixes[0]))
    candidates = [preferred]
    for ext in suffixes:
        candidates.extend([directory / ("H3_%05d%s" % (index, ext)), directory / ("clip_%05d%s" % (index, ext))])
        candidates.extend(sorted(directory.glob("*_%05d%s" % (index, ext))))
    for path in candidates:
        if path.is_file():
            return path
    return preferred


def _paths(plan, index: int, output_name="", video_format="mp4", for_write=False,
           for_context=False, context_stage=2):
    base = Path(plan["project_dir"])
    clips = base / ((CONTEXT_STAGE1_DIR_NAME if int(context_stage) == 1 else CONTEXT_DIR_NAME)
                    if for_context else "clips")
    cache = base / ("cache_stage1" if for_context and int(context_stage) == 1 else "cache")
    if for_write:
        stem = _output_filename(output_name)
        ext = "." + str(video_format or "mp4").lower().lstrip(".")
        if ext not in VIDEO_EXTENSIONS:
            ext = ".mp4"
        return clips / ("%s_%05d%s" % (stem, index, ext)), cache / ("%s_%05d.safetensors" % (stem, index))
    video = _indexed_file(clips, index, VIDEO_EXTENSIONS, output_name)
    if for_context and not video.exists():
        # Projects written before the context/clip split remain resumable.
        video = _indexed_file(base / "clips", index, VIDEO_EXTENSIONS, output_name)
    latent = _indexed_file(cache, index, ".safetensors", output_name)
    if not latent.exists():
        latent = _indexed_file(base / "latents", index, ".safetensors", output_name)
    # A project created before the h3_project/<folder> layout may still hold
    # the previous clip/cache pair. Read it only as a continuation fallback;
    # all new writes stay in the current project directory.
    legacy_dir = str(plan.get("legacy_project_dir") or "").strip()
    if legacy_dir and (not video.exists() or not latent.exists()):
        old_base = Path(legacy_dir)
        old_video = _indexed_file(old_base / "clips", index, VIDEO_EXTENSIONS, output_name)
        old_latent = _indexed_file(old_base / "cache", index, ".safetensors", output_name)
        if not old_latent.exists():
            old_latent = _indexed_file(old_base / "latents", index, ".safetensors", output_name)
        if not video.exists() and old_video.exists():
            video = old_video
        if not latent.exists() and old_latent.exists():
            latent = old_latent
    return video, latent


def _json_path(plan, name):
    base = Path(plan["project_dir"])
    modern = base / "json" / name
    legacy = base / name
    return modern if modern.exists() or not legacy.exists() else legacy


def _state_path(plan):
    return _json_path(plan, "state.json")


def _runtime_plan(plan, output_root=None):
    """Return a plan copy whose project directory remains stable across queued runs."""
    root = str(plan.get("output_root") or "h3_projects").strip()
    runtime = dict(plan)
    runtime["output_root"] = root
    existing = str(plan.get("project_dir") or "").strip()
    runtime["project_dir"] = existing if existing else str(_safe_project_dir(plan["project_id"], root))
    return runtime


def _load_context_video(path: Path, max_frames=39):
    if av is None:
        raise RuntimeError("PyAV is required to load H3 context videos")
    # ``max_frames=None`` is used only by disk assembly diagnostics/fallbacks;
    # normal context loading keeps a bounded tail to avoid retaining a whole
    # long project in memory.
    tail = deque(maxlen=(None if max_frames is None else max(1, int(max_frames))))
    with av.open(str(path), "r") as container:
        streams = tuple(container.streams.video)
        if not streams:
            raise ValueError("上下文文件不含视频流，不能用于画面接续：%s。请重新生成该片段。" % path)
        stream = streams[0]
        for frame in container.decode(stream):
            tail.append(frame.to_ndarray(format="rgb24"))
    if not tail:
        raise ValueError("Context video has no frames: %s" % path)
    return torch.from_numpy(__import__("numpy").stack(list(tail))).float() / 255.0


def _color_match_to_reference(images, reference, blend=0.75, scene_cut_protection=True,
                              scene_cut_threshold=0.18, residual_strength=0.2, return_info=False):
    """Apply one conservative RGB color transform to a whole clip.

    Statistics are measured from the tail of both clips, but the same transform
    is applied to every frame so the correction cannot introduce frame flicker.
    """
    info = {"scene_cut": False, "correction_applied": False, "global_offset": [0.0, 0.0, 0.0], "gain": [1.0, 1.0, 1.0], "residual_max": 0.0}
    if not torch.is_tensor(images) or not torch.is_tensor(reference):
        return (images, info) if return_info else images
    if images.ndim != 4 or reference.ndim != 4 or images.shape[-1] < 3 or reference.shape[-1] < 3:
        return (images, info) if return_info else images
    source = images[..., :3].float()
    ref = reference[..., :3].float().to(device=source.device)
    ref = torch.nn.functional.interpolate(
        ref.permute(0, 3, 1, 2), size=source.shape[1:3], mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)
    source_sample = source[-min(39, source.shape[0]):].reshape(-1, 3)
    ref_sample = ref[-min(39, ref.shape[0]):].reshape(-1, 3)
    head_n = min(5, source.shape[0], ref.shape[0])
    src_head = source[:head_n]
    ref_tail = ref[-head_n:]
    # A large luminance/chroma and edge mismatch is a deliberate scene cut,
    # not drift. Do not force the new scene toward the previous one.
    src_luma = (src_head * torch.tensor([0.2126, 0.7152, 0.0722], device=source.device)).sum(-1).mean()
    ref_luma = (ref_tail * torch.tensor([0.2126, 0.7152, 0.0722], device=source.device)).sum(-1).mean()
    color_delta = (src_head.mean((0, 1, 2)) - ref_tail.mean((0, 1, 2))).abs().mean()
    scene_cut = bool(scene_cut_protection and max(float((src_luma - ref_luma).abs()), float(color_delta)) > float(scene_cut_threshold))
    info["scene_cut"] = scene_cut
    if scene_cut:
        return (images, info) if return_info else images
    source_mean = source_sample.mean(dim=0)
    ref_mean = ref_sample.mean(dim=0)
    source_std = source_sample.std(dim=0, unbiased=False).clamp_min(1e-3)
    ref_std = ref_sample.std(dim=0, unbiased=False)
    gain = (ref_std / source_std).clamp(0.75, 1.33)
    offset = ref_mean - source_mean * gain
    corrected = source * gain + offset
    corrected = source + (corrected - source) * float(blend)
    if residual_strength > 0:
        # A tiny independent per-frame correction removes encoder/VAE drift
        # without recursively propagating errors from one frame to the next.
        target_mean = ref_sample.mean(dim=0)
        residual_limit = 0.02 * float(max(0.0, min(1.0, residual_strength)))
        residual = target_mean - corrected.mean(dim=(1, 2))
        residual = torch.clamp(residual, -residual_limit, residual_limit)
        corrected = corrected + residual[:, None, None, :]
        info["residual_max"] = float(residual.abs().max().detach().cpu())
    corrected = corrected.clamp(0.0, 1.0)
    info["correction_applied"] = True
    info["global_offset"] = [float(x) for x in offset.detach().cpu()]
    info["gain"] = [float(x) for x in gain.detach().cpu()]
    if images.shape[-1] > 3:
        corrected = torch.cat((corrected, images[..., 3:]), dim=-1)
    return (corrected, info) if return_info else corrected


def _color_reference_path(plan, segment_index, color_correction, output_name, video_format):
    """Find the saved clip used as the selected color anchor."""
    anchor_index = 1 if color_correction == "匹配首段" else max(1, int(segment_index) - 1)
    preferred, _ = _paths(plan, anchor_index, output_name, video_format, for_context=True)
    if preferred.is_file():
        return preferred
    fallback, _ = _paths(plan, anchor_index, for_context=True)
    return fallback if fallback.is_file() else None


def _load_av_latent(path: Path):
    if st_load is None:
        raise RuntimeError("safetensors is required for H3 AV latent caches")
    values = st_load(str(path), device="cpu")
    if "video" not in values or "audio" not in values:
        raise ValueError("Not an H3 AV latent cache: %s" % path)
    return {"samples": [values["video"], values["audio"]]}


def _av_latent_parts(value):
    """Return video/audio tensors from both old list and ComfyUI NestedTensor layouts."""
    if not isinstance(value, dict):
        return None
    samples = value.get("samples")
    # ComfyUI v0.31 stores MiniMax H3's joint latent as NestedTensor.
    tensors = getattr(samples, "tensors", None)
    if isinstance(tensors, (list, tuple)) and len(tensors) >= 2:
        return tensors[0], tensors[1]
    unbind = getattr(samples, "unbind", None)
    if callable(unbind):
        try:
            tensors = list(unbind())
        except Exception:
            tensors = None
        if isinstance(tensors, list) and len(tensors) >= 2:
            return tensors[0], tensors[1]
    if isinstance(samples, (list, tuple)) and len(samples) >= 2:
        return samples[0], samples[1]
    # Accept cache-like dictionaries produced by older/custom nodes too.
    video = value.get("video")
    audio = value.get("audio")
    if torch.is_tensor(video) and torch.is_tensor(audio):
        return video, audio
    # A few ComfyUI wrappers expose the audio stream beside samples.
    audio = value.get("audio_samples")
    if audio is None:
        audio = value.get("audio_latent")
    if torch.is_tensor(samples) and torch.is_tensor(audio):
        return samples, audio
    return None


def _h3_av_container(video, audio):
    """Create an AV latent in the representation supported by this core.

    ComfyUI 0.31+ packs H3 video/audio streams as ``NestedTensor``.  The
    integrated 0.30 compatibility path expects the older two-item list.
    Centralising this conversion prevents a new context feature from making
    the entire plugin unloadable on the older supported core.
    """
    if _h3_nested_tensor is not None:
        return _h3_nested_tensor.NestedTensor((video, audio))
    return [video, audio]


def _h3_sigmas(model, scheduler, steps, denoise):
    """Build the same sigma schedule as ComfyUI's BasicScheduler node."""
    steps = max(1, int(steps))
    denoise = max(0.0, min(1.0, float(denoise)))
    if denoise <= 0.0:
        return torch.empty((0,), dtype=torch.float32)
    total_steps = steps if denoise >= 1.0 else max(steps, int(steps / denoise))
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), str(scheduler), total_steps
    ).cpu()
    return sigmas[-(steps + 1):]


def _extend_intermediate_sigmas(sigmas, steps, start_at_sigma, end_at_sigma, spacing):
    """Run ComfyUI's ExtendIntermediateSigmas algorithm unchanged.

    This intentionally mirrors ``comfy_extras.nodes_custom_sampler``.  The
    dual sampler only supplies the stage's already-generated base schedule;
    all four controls retain the core node's exact meaning and output shape.
    """
    if start_at_sigma < 0:
        start_at_sigma = float("inf")

    interpolator = {
        "linear": lambda x: x,
        "cosine": lambda x: torch.sin(x * math.pi / 2),
        "sine": lambda x: 1 - torch.cos(x * math.pi / 2),
    }[spacing]

    x = torch.linspace(0, 1, steps + 1, device=sigmas.device)[1:-1]
    computed_spacing = interpolator(x)

    extended_sigmas = []
    for i in range(len(sigmas) - 1):
        sigma_current = sigmas[i]
        sigma_next = sigmas[i + 1]
        extended_sigmas.append(sigma_current)

        if end_at_sigma <= sigma_current <= start_at_sigma:
            interpolated_steps = computed_spacing * (sigma_next - sigma_current) + sigma_current
            extended_sigmas.extend(interpolated_steps.tolist())

    if len(sigmas) > 0:
        extended_sigmas.append(sigmas[-1])

    return torch.FloatTensor(extended_sigmas)


_AUDIO_SAMPLING_SIGMAS_MARKER = "_h3_auto_director_audio_sampling_base"
_AUDIO_SAMPLING_SIGMAS_INFO = "_h3_auto_director_audio_sampling_info"


def _audio_sampling_from_base_sigmas(value):
    """Return audio sampling metadata only for this node's raw Sigma output.

    A core Sigma node produces a fresh tensor and intentionally drops this
    marker. That makes a schedule expanded by the core node an ordinary,
    user-authored external schedule, while a direct connection keeps the
    dual-sampler stage controls authoritative.
    """
    if not torch.is_tensor(value) or not bool(getattr(value, _AUDIO_SAMPLING_SIGMAS_MARKER, False)):
        return None
    info = getattr(value, _AUDIO_SAMPLING_SIGMAS_INFO, None)
    return dict(info) if isinstance(info, dict) else None


def _is_conditioning_entry(value):
    """Return whether *value* is one ComfyUI CONDITIONING entry.

    Standard nodes return ``[[embedding, metadata], ...]``.  Prompt-cache and
    compatibility nodes may return the single entry directly as
    ``[embedding, metadata]`` (or use a tuple for the pair).  Treat both forms
    identically so a valid positive condition is not mistaken for an empty
    list.
    """
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        # The embedding is normally a tensor, but compatibility/cache nodes
        # may wrap it in a tensor-like object.  The metadata dict is the
        # reliable discriminator for a CONDITIONING entry.
        and value[0] is not None
        and isinstance(value[1], dict)
    )


def _conditioning_entries(value):
    """Normalize standard and single-entry CONDITIONING values."""
    if value is None:
        return []
    if _is_conditioning_entry(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _has_positive_conditioning(value):
    """Check for at least one usable positive CONDITIONING entry."""
    return any(_is_conditioning_entry(entry) for entry in _conditioning_entries(value))


def _without_auto_director_audio_context(conditioning):
    """Remove only Auto Director's audio continuation guides.

    H3's native packed layout reserves rows from the metadata blocks while the
    model consumes the actual latent tensors.  A cached conditioning block can
    retain an old audio window after the current segment's audio length has
    changed.  Removing the private continuation audio is a safe recovery path:
    video keyframes and user-provided audio references remain untouched.
    """
    changed = False
    prepared = []
    for entry in _conditioning_entries(conditioning):
        if not _is_conditioning_entry(entry):
            prepared.append(entry)
            continue
        values = entry[1].copy()
        keyframes = values.get("minimax_keyframes")
        if isinstance(keyframes, (list, tuple)):
            updated = []
            for keyframe in keyframes:
                if not isinstance(keyframe, dict):
                    updated.append(keyframe)
                    continue
                item = dict(keyframe)
                if item.get(_NATIVE_CONTEXT_KEY) and item.get("audio_latent") is not None:
                    item.pop("audio_latent", None)
                    changed = True
                updated.append(item)
            values["minimax_keyframes"] = updated
        refs = values.get("minimax_refs")
        if isinstance(refs, (list, tuple)):
            updated_refs = []
            for ref in refs:
                if not isinstance(ref, dict):
                    updated_refs.append(ref)
                    continue
                if ("motion_context_audio_end_frame" in ref
                        or "h3_auto_director_legacy_audio_end_frame" in ref):
                    changed = True
                    continue
                updated_refs.append(ref)
            values["minimax_refs"] = updated_refs
        prepared.append([entry[0], values, *entry[2:]])
    return prepared if changed else None


def _dual_sample(model, conditioning, latent, sampler_name, scheduler, steps, denoise, seed,
                 enable_preview=False, sigmas=None, extend_sigmas=False,
                 extend_steps=2, extend_start_at_sigma=-1.0,
                 extend_end_at_sigma=12.0, extend_spacing="linear"):
    """Run one positive-only H3 sampling pass, matching BasicGuider semantics."""
    conditioning = _conditioning_entries(conditioning)
    if not _has_positive_conditioning(conditioning):
        raise ValueError("双采样需要有效的正向条件")
    from comfy_extras.nodes_custom_sampler import Guider_Basic
    # Always install the layout refresh so cached conditioning cannot retain a
    # previous segment's audio row count (for example 470 vs 396).
    model = ensure_h3_layout_refresh(model)
    guider = Guider_Basic(model)
    guider.set_conds(conditioning)
    sampler = comfy.samplers.sampler_object(str(sampler_name))
    base_sampling_info = _audio_sampling_from_base_sigmas(sigmas)
    if sigmas is None or base_sampling_info is not None:
        sigmas = _h3_sigmas(model, scheduler, steps, denoise)
        if bool(extend_sigmas):
            sigmas = _extend_intermediate_sigmas(
                sigmas, int(extend_steps), float(extend_start_at_sigma),
                float(extend_end_at_sigma), str(extend_spacing)
            )
            steps = int(sigmas.numel() - 1)
            LOG.info("H3 Auto Director: 使用原版 ExtendIntermediateSigmas 扩展调度（%d 步）", steps)
        elif base_sampling_info is not None:
            LOG.info("H3 Auto Director: 音频采样切换基础 SIGMAS 已直接连接，保留阶段调度（%d 步）", steps)
    else:
        if not torch.is_tensor(sigmas):
            try:
                sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
            except Exception as exc:
                raise ValueError("外部 Sigmas 必须是有效的 SIGMAS 张量") from exc
        sigmas = sigmas.flatten().contiguous()
        if sigmas.numel() < 2:
            raise ValueError("外部 Sigmas 至少需要包含 2 个值")
        steps = int(sigmas.numel() - 1)
        LOG.info("H3 Auto Director: 使用外部 Sigmas 调度（%d 步）", steps)
    if sigmas.numel() == 0:
        return dict(latent)
    latent_image = latent["samples"]
    noise = comfy.sample.prepare_noise(latent_image, int(seed), latent.get("batch_index"))
    denoise_mask = latent.get("noise_mask")
    if denoise_mask is not None:
        mask_parts = _av_latent_parts({"samples": denoise_mask})
        if mask_parts is not None and torch.is_tensor(mask_parts[0]):
            LOG.info(
                "H3 Auto Director: 采样器接收上下文 denoise_mask，视频 shape=%s，范围=[%.3f, %.3f]",
                tuple(int(value) for value in mask_parts[0].shape),
                float(mask_parts[0].amin().item()), float(mask_parts[0].amax().item()),
            )
        else:
            LOG.info("H3 Auto Director: 采样器接收上下文 denoise_mask（类型=%s）", type(denoise_mask).__name__)
    callback = None
    if bool(enable_preview):
        try:
            import latent_preview
            callback = latent_preview.prepare_callback(model, int(steps))
        except Exception as exc:
            LOG.warning("H3 Auto Director: 新版采样预览初始化失败，继续采样：%s", exc)
    try:
        samples = guider.sample(noise, latent_image, sampler, sigmas,
                                denoise_mask=denoise_mask, callback=callback, seed=int(seed))
    except RuntimeError as exc:
        # Retry once with the private Auto Director audio guide removed. This
        # preserves video continuation and all user audio references when a
        # legacy graph or an incompatible core still reports a packed-layout
        # audio-row mismatch.
        message = str(exc)
        if not ("expanded size" in message and "5376" in message):
            raise
        repaired = _without_auto_director_audio_context(conditioning)
        if repaired is None:
            raise
        LOG.warning(
            "H3 Auto Director: 检测到音频上下文行数不匹配，已移除自动导演音频上下文并保留视频上下文后重试"
        )
        retry_guider = Guider_Basic(model)
        retry_guider.set_conds(repaired)
        retry_noise = comfy.sample.prepare_noise(latent_image, int(seed), latent.get("batch_index"))
        samples = retry_guider.sample(retry_noise, latent_image, sampler, sigmas,
                                      denoise_mask=denoise_mask, callback=callback,
                                      seed=int(seed))
    result = dict(latent)
    result["samples"] = samples
    return result


def _apply_audio_sampling_config(model, audio_sampling, stage_label):
    """Apply the optional audio-sampling configuration without owning SIGMAS.

    The dual sampler remains the sole owner of scheduler, step count and
    denoise.  Keeping this separate prevents a fixed schedule from silently
    overriding the stage controls in a workflow.
    """
    if audio_sampling is None:
        return model
    if not isinstance(audio_sampling, dict):
        raise ValueError("音频采样配置必须来自‘H3 自动导演｜音频采样切换’节点")
    try:
        sampling_mode = audio_sampling["sampling_mode"]
        shift_video = float(audio_sampling["shift_video"])
        shift_audio = float(audio_sampling["shift_audio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("音频采样配置不完整：需要采样方法、视频偏移和音频偏移") from exc
    LOG.info("H3 Auto Director: 对%s应用音频采样配置", stage_label)
    return apply_h3_sampling(model, sampling_mode, shift_video, shift_audio)


def _mark_motion_context(conditioning):
    """Tag only the conditioning emitted by this adapter, not user references."""
    return node_helpers.conditioning_set_values(conditioning, {_MOTION_CONTEXT_MARKER: True})


def _native_h3_add_guide_supported():
    """Whether this ComfyUI core natively accepts arbitrary H3 guide frames.

    ComfyUI v0.31's H3 PackedLayout accepts arbitrary ``resolved_frame_index``
    and audio latents directly on a keyframe.  Older cores restrict anchors to
    first/last frame and require the external Motion Context layout patch.
    Probe the constructor instead of relying on a version string because many
    portable ComfyUI distributions backport only part of the H3 changes.
    """
    try:
        module = importlib.import_module("comfy.ldm.minimax.model")
        parameters = inspect.signature(module.PackedLayout.__init__).parameters
        return (hasattr(module, "FRAME_PER_TOKEN")
                and "frame_count" not in parameters)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _h3_pixel_frames(latent_t):
    """Return H3 pixel frames represented by a video latent's temporal axis."""
    try:
        module = importlib.import_module("comfy.ldm.minimax.model")
        frame_per_token = module.FRAME_PER_TOKEN
    except (ImportError, AttributeError):
        frame_per_token = (1, 4, 4, 4, 4)
    return sum(frame_per_token[index % len(frame_per_token)] for index in range(int(latent_t)))


def _h3_context_run(context_length):
    """Snap the requested context to a distinct H3 VAE temporal run."""
    return next((value for value in (39, 22, 5, 1) if value <= int(context_length)), 1)


def _h3_audio_tail_from_latent(context_latent, frame_count):
    """Read a context audio tail without depending on Motion Context nodes."""
    parts = _av_latent_parts(context_latent)
    if parts is None or not torch.is_tensor(parts[0]) or not torch.is_tensor(parts[1]):
        raise ValueError("上下文 AV latent 无效，无法读取音频上下文")
    video, audio = parts
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError("H3 上下文 latent 格式无效")
    total_steps = int(audio.shape[-1])
    wanted_steps = max(1, int(round(int(frame_count) / FPS * 40.0)))
    actual_steps = min(wanted_steps, total_steps)
    if actual_steps < 1:
        raise ValueError("上下文音频 latent 为空")
    frame_total = _h3_pixel_frames(int(video.shape[2]))
    overhang = float(total_steps) - (5.0 / 3.0) * float(frame_total)
    if not 0.0 <= overhang < 1.0:
        overhang = 0.0
    return audio[:1, ..., -actual_steps:].clone(), actual_steps, overhang


def _nearest_multiple(value, multiple=32):
    """Round a positive dimension to its nearest H3 canvas multiple."""
    multiple = max(1, int(multiple))
    # Use conventional half-up rounding so backend values match the browser's
    # Math.round behavior (for example, 80 -> 96 rather than Python's 64).
    return max(multiple, int(math.floor(float(value) / multiple + 0.5)) * multiple)


def _prepare_dual_sampling_conditioning(conditioning, strip_motion_context=False,
                                        preserve_motion_context_marker=False):
    """Remove internal tags and, for stage two, only Motion Context payloads.

    The stage-one sampler is responsible for joining a project segment to its
    predecessor.  Stage two starts from the upscaled stage-one latent, so
    pinning the predecessor a second time competes with that latent and
    effectively applies continuation twice.  User supplied ``minimax_refs``
    must remain available in both stages.
    """
    prepared = []
    for entry in _conditioning_entries(conditioning):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            prepared.append(entry)
            continue
        values = entry[1].copy()
        if not preserve_motion_context_marker:
            values.pop("h3_stage2_context_latent", None)
        is_motion_context = bool(values.get(_MOTION_CONTEXT_MARKER, False))
        if not preserve_motion_context_marker:
            values.pop(_MOTION_CONTEXT_MARKER, None)
        if strip_motion_context and is_motion_context:
            keyframes = values.get("minimax_keyframes")
            if isinstance(keyframes, (list, tuple)):
                kept_keyframes = [item for item in keyframes if not (
                    isinstance(item, dict) and (
                        item.get(_NATIVE_CONTEXT_KEY)
                        or item.get("h3_auto_director_legacy_frame_index") is not None
                    )
                )]
                if kept_keyframes:
                    values["minimax_keyframes"] = kept_keyframes
                else:
                    values.pop("minimax_keyframes", None)
                    values.pop("minimax_frame_count", None)
            refs = values.get("minimax_refs")
            if isinstance(refs, (list, tuple)):
                # The adapter marks its audio continuation ref with the
                # Motion Context timeline field. Leave all normal user refs.
                retained = [ref for ref in refs if not (
                    isinstance(ref, dict) and (
                        "motion_context_audio_end_frame" in ref
                        or "h3_auto_director_legacy_audio_end_frame" in ref
                    )
                )]
                if retained:
                    values["minimax_refs"] = retained
                else:
                    values.pop("minimax_refs", None)
            values.pop(_NATIVE_CONTEXT_KEY, None)
        prepared.append([entry[0], values, *entry[2:]])
    return prepared


def _resize_h3_context_latent(latent, target_h, target_w):
    """Resize a visual Motion Context latent to the current H3 latent grid.

    The first and second passes of dual sampling intentionally use different
    spatial grids.  Motion Context keyframes are encoded on the first-pass
    grid, while the second pass consumes the upscaled grid.  Passing the old
    keyframe tensor through unchanged makes ``PackedLayout.img_update`` expect
    a different number of patch rows than ``cond_video_rows`` provides, which
    aborts sampling with a shape-mismatch error.  Keep the temporal context
    steps intact and resize only H/W; these are conditioning rows, not the
    denoised video itself.
    """
    if not torch.is_tensor(latent) or latent.ndim != 5:
        return latent
    target_h, target_w = int(target_h), int(target_w)
    if target_h <= 0 or target_w <= 0 or tuple(latent.shape[-2:]) == (target_h, target_w):
        return latent
    b, c, t, h, w = latent.shape
    if h == target_h and w == target_w:
        return latent
    # Interpolate each temporal latent independently so no context timestep is
    # invented or dropped during the resolution transition.
    flat = latent.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    flat = torch.nn.functional.interpolate(flat, size=(target_h, target_w),
                                            mode="bilinear", align_corners=False)
    return flat.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).to(
        device=latent.device, dtype=latent.dtype
    )


def _prepare_stage2_conditioning(conditioning, latent, use_context, context_source=None):
    """Prepare second-pass conditioning against the current pass's grid.

    ``latent`` is always the current second-pass input.  ``context_source``
    is the predecessor's final stage-two AV latent and must never be used as
    the target grid merely because it is connected to the optional context
    socket.
    """
    parts = _av_latent_parts(latent)
    if not parts or not torch.is_tensor(parts[0]) or parts[0].ndim != 5:
        return _prepare_dual_sampling_conditioning(
            conditioning, strip_motion_context=not bool(use_context)
        )
    target_video = parts[0]

    def adapt_keyframe_grids(prepared):
        """Keep native guide keyframes on the active second-pass grid.

        A cached first-pass conditioning can retain guide latents encoded at
        the first-pass resolution.  PackedLayout uses the target grid for
        keyframe rows, so leaving those latents unchanged makes its boolean
        update mask longer than the actual condition rows.
        """
        adapted = []
        for entry in prepared:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
                adapted.append(entry)
                continue
            values = entry[1].copy()
            keyframes = values.get("minimax_keyframes")
            if isinstance(keyframes, (list, tuple)):
                updated = []
                for keyframe in keyframes:
                    if not isinstance(keyframe, dict):
                        updated.append(keyframe)
                        continue
                    item = dict(keyframe)
                    if torch.is_tensor(item.get("latent")):
                        item["latent"] = _resize_h3_context_latent(
                            item["latent"], target_video.shape[-2], target_video.shape[-1]
                        )
                    updated.append(item)
                values["minimax_keyframes"] = updated
            adapted.append([entry[0], values, *entry[2:]])
        return adapted

    if not bool(use_context):
        return adapt_keyframe_grids(
            _prepare_dual_sampling_conditioning(conditioning, strip_motion_context=True)
        )
    source_parts = _av_latent_parts(context_source) if context_source is not None else None
    # Never use the current segment's first-pass latent as a predecessor.
    # Doing so makes the fallback appear to apply context while actually
    # conditioning the second pass on the segment being generated now.
    source_video, source_audio = source_parts if source_parts is not None else (None, None)
    prepared = []
    replaced_context = False
    for entry in _prepare_dual_sampling_conditioning(
        conditioning, preserve_motion_context_marker=True
    ):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            prepared.append(entry)
            continue
        values = entry[1].copy()
        if values.get(_MOTION_CONTEXT_MARKER):
            replaced_context = True
            saved_stage2 = values.pop("h3_stage2_context_latent", None)
            saved_parts = _av_latent_parts(saved_stage2)
            if saved_parts is not None:
                source_video, source_audio = saved_parts
            keyframes = values.get("minimax_keyframes")
            if isinstance(keyframes, (list, tuple)):
                source_video_resized = _resize_h3_context_latent(
                    source_video, target_video.shape[-2], target_video.shape[-1]
                ) if torch.is_tensor(source_video) else None
                source_steps = int(source_video_resized.shape[2]) if torch.is_tensor(source_video_resized) else 0
                # Only replace Auto Director's own continuation keyframe.
                # User-inserted Guide keyframes are independent references and
                # must not be overwritten by the predecessor video.
                step_counts = [
                    int(item.get("latent").shape[2])
                    if isinstance(item, dict) and torch.is_tensor(item.get("latent"))
                    and item["latent"].ndim == 5 else 0
                    for item in keyframes
                ]
                context_flags = [
                    isinstance(item, dict) and bool(
                        item.get(_NATIVE_CONTEXT_KEY)
                        or item.get("h3_auto_director_legacy_frame_index") is not None
                    )
                    for item in keyframes
                ]
                # Keep the complete context window so its temporal layout
                # still matches SaveSegment's 22-frame crop.  The entire
                # Auto Director keyframe must come from one source pass:
                # mixing six stage-one tokens with one stage-two token creates
                # a discontinuity inside the guide itself and appears as a
                # soft/jumping first few frames. User-inserted guide keyframes
                # remain untouched below.
                context_run = int(values.get("h3_auto_director_context_run", FRAME_CONTEXT_DEFAULT) or FRAME_CONTEXT_DEFAULT)
                context_run = _h3_context_run(context_run)
                context_step_count = {1: 1, 5: 2, 22: 7, 39: 12}[context_run]
                required_steps = sum(step_counts[index]
                                     for index, is_context in enumerate(context_flags)
                                     if is_context)
                if required_steps:
                    LOG.info(
                        "H3 Auto Director: 二采使用上一段最终结果的完整视觉上下文（%d 个 latent 步）",
                        int(required_steps),
                    )
                source_tail = (source_video_resized[:, :, -context_step_count:]
                               if context_step_count > 0 and source_steps >= context_step_count else None)
                resized = []
                cursor = 0
                for index, keyframe in enumerate(keyframes):
                    item = dict(keyframe)
                    item_steps = step_counts[index]
                    if source_tail is not None and context_flags[index] and item_steps > 0:
                        # The normal context keyframe occupies frame 0 and
                        # receives only the exact predecessor context window.
                        # No extra keyframe is placed at the cut: temporal
                        # tokens span multiple pixel frames and such an anchor
                        # would overlap the first generated frames. Legacy
                        # mode stores the window as one-token keyframes, so
                        # retain their chronological order with a cursor.
                        frame_index = item.get("resolved_frame_index")
                        if frame_index is None:
                            frame_index = item.get("h3_auto_director_legacy_frame_index", 0)
                        if int(frame_index or 0) >= context_run:
                            item["latent"] = source_video_resized[:, :, -item_steps:]
                        elif item_steps > 1:
                            item["latent"] = source_tail[:, :, -item_steps:]
                        else:
                            end = min(int(source_tail.shape[2]), cursor + item_steps)
                            start = max(0, end - item_steps)
                            item["latent"] = source_tail[:, :, start:end]
                            cursor = end
                    elif torch.is_tensor(item.get("latent")):
                        item["latent"] = _resize_h3_context_latent(
                            item.get("latent"), target_video.shape[-2], target_video.shape[-1]
                        )
                    resized.append(item)
                values["minimax_keyframes"] = resized
                # The predecessor tail is an exact continuity anchor in the
                # second pass. H3's default 0.999 visual-condition noise is
                # useful for ordinary references, but it can soften the
                # first decoded frames when applied to this boundary guide.
                values["visual_cond_noise_aug"] = 1.0
            refs = values.get("minimax_refs")
            if isinstance(refs, (list, tuple)) and torch.is_tensor(source_audio):
                # Only replace the Motion Context audio ref. User-supplied
                # audio references must keep their own latent. The previous
                # implementation also sliced ``[:, :, -ref_t:]``; for H3's
                # [B, C, stereo, time] audio latent that indexes the stereo
                # axis and leaves the full time axis intact. The layout then
                # reserved 2*ref_t rows while the payload supplied 2*total_t
                # rows (the 470-vs-74 crash seen on the second segment).
                updated_refs = []
                for ref in refs:
                    item = dict(ref)
                    ref_t = int(item.get("ref_audio_t", 0) or 0)
                    is_context_audio = ("motion_context_audio_end_frame" in item
                                        or "h3_auto_director_legacy_audio_end_frame" in item)
                    if is_context_audio and ref_t > 0:
                        if item.get("audio_latent") is None:
                            # Never leave a positive ref_audio_t without a
                            # payload: PackedLayout would reserve rows that
                            # _cond_audio_rows cannot fill.
                            item["ref_audio_t"] = 0
                            item.pop("motion_context_audio_end_frame", None)
                        elif source_audio.ndim != 4:
                            LOG.warning("H3 Auto Director: 二采上下文音频 latent 维度异常，跳过音频上下文：%s", tuple(source_audio.shape))
                            item.pop("audio_latent", None)
                            item["ref_audio_t"] = 0
                            item.pop("motion_context_audio_end_frame", None)
                        else:
                            available_t = int(source_audio.shape[-1])
                            actual_t = min(ref_t, available_t)
                            if actual_t <= 0:
                                item.pop("audio_latent", None)
                                item["ref_audio_t"] = 0
                                item.pop("motion_context_audio_end_frame", None)
                            else:
                                # Time is the final dimension: [B, C, 2, T].
                                item["audio_latent"] = source_audio[..., -actual_t:]
                                item["ref_audio_t"] = actual_t
                                if actual_t != ref_t:
                                    # Keep the timeline marker valid for the
                                    # shortened window; the end coordinate is
                                    # unchanged, only its width is reduced.
                                    LOG.warning(
                                        "H3 Auto Director: 二采上下文音频仅有 %d 个 latent 步，原请求 %d，已截取可用长度",
                                        actual_t, ref_t,
                                    )
                    updated_refs.append(item)
                values["minimax_refs"] = updated_refs
            # Native H3 guide audio lives on a keyframe rather than a ref.
            # Replace only the Auto Director-marked guide with the separately
            # saved final (stage-two) context audio; user-added guide audio is
            # deliberately untouched.
            if torch.is_tensor(source_audio) and source_audio.ndim == 4:
                native_keyframes = values.get("minimax_keyframes")
                if isinstance(native_keyframes, (list, tuple)):
                    # Native keyframes loaded from a cached conditioning can
                    # carry the whole previous segment's audio (for example
                    # 198/235 latent steps). A continuation guide must contain
                    # only the configured context tail on H3's 40 Hz axis;
                    # otherwise its packed rows no longer describe the current
                    # segment and the audio transition is over-conditioned.
                    context_audio_steps = max(
                        1, int(round(float(context_run) / FPS * 40.0))
                    )
                    context_audio_steps = min(
                        context_audio_steps, int(source_audio.shape[-1])
                    )
                    updated_keyframes = []
                    for keyframe in native_keyframes:
                        item = dict(keyframe)
                        old_audio = item.get("audio_latent")
                        if (item.get(_NATIVE_CONTEXT_KEY) and torch.is_tensor(old_audio)
                                and old_audio.ndim == 4):
                            actual_t = min(context_audio_steps, int(old_audio.shape[-1]))
                            if actual_t > 0:
                                item["audio_latent"] = source_audio[..., -actual_t:].to(
                                    device=target_video.device, dtype=target_video.dtype)
                        updated_keyframes.append(item)
                    values["minimax_keyframes"] = updated_keyframes
            # The keyframe timeline is still the current segment's timeline;
            # the frame count from the first pass is valid for the same length.
            # Keep the marker out of the model payload after this pass; it is
            # only an internal routing flag for this preparation step.
            values.pop(_MOTION_CONTEXT_MARKER, None)
            prepared.append([entry[0], values, *entry[2:]])
            LOG.info(
                "H3 Auto Director: 二采已应用上一段最终二采上下文：视频尾部=%s，音频=%s，目标 latent=%s",
                tuple(int(v) for v in source_video.shape) if torch.is_tensor(source_video) else "无",
                "开启" if torch.is_tensor(source_audio) else "关闭",
                tuple(int(v) for v in target_video.shape),
            )
        else:
            # A separately cached/encoded positive condition may not carry
            # Auto Director's private Motion Context marker.  It is still a
            # valid condition and must remain available for stage two; the
            # fallback guide injection below can add context independently.
            prepared.append([entry[0], values, *entry[2:]])
    # Some third-party prompt-cache nodes rebuild CONDITIONING and drop
    # Auto Director's private marker.  The guide payload itself is still
    # authoritative: keep it and adapt its spatial grid for stage two before
    # considering any fallback source.
    if not replaced_context and prepared:
        retained_context = False
        retained_source = _resize_h3_context_latent(
            source_video, target_video.shape[-2], target_video.shape[-1]
        ) if torch.is_tensor(source_video) else None
        for index, entry in enumerate(prepared):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
                continue
            values = entry[1].copy()
            keyframes = values.get("minimax_keyframes")
            if not isinstance(keyframes, (list, tuple)):
                continue
            updated_keyframes = []
            has_context_keyframe = False
            for keyframe in keyframes:
                item = dict(keyframe)
                is_context_keyframe = bool(
                    item.get(_NATIVE_CONTEXT_KEY)
                    or item.get("h3_auto_director_legacy_frame_index") is not None
                )
                if is_context_keyframe and torch.is_tensor(item.get("latent")):
                    # Prompt-vector caches may drop the private entry marker,
                    # but the keyframe itself remains identifiable. Apply the
                    # same complete-window replacement in that case.
                    if retained_source is not None and retained_source.shape[2] > 0:
                        item_steps = int(item["latent"].shape[2])
                        if int(retained_source.shape[2]) >= item_steps:
                            item["latent"] = retained_source[:, :, -item_steps:]
                        else:
                            # Very short legacy caches can be resumed too;
                            # repeat their oldest available token rather than
                            # changing the keyframe row count and breaking
                            # PackedLayout's condition payload alignment.
                            missing = item_steps - int(retained_source.shape[2])
                            prefix = retained_source[:, :, :1].repeat(1, 1, missing, 1, 1)
                            item["latent"] = torch.cat((prefix, retained_source), dim=2)
                    else:
                        item["latent"] = _resize_h3_context_latent(
                            item["latent"], target_video.shape[-2], target_video.shape[-1]
                        )
                    has_context_keyframe = True
                updated_keyframes.append(item)
            if has_context_keyframe:
                values["minimax_keyframes"] = updated_keyframes
                values["visual_cond_noise_aug"] = 1.0
                values.pop(_MOTION_CONTEXT_MARKER, None)
                values.pop("h3_stage2_context_latent", None)
                prepared[index] = [entry[0], values, *entry[2:]]
                retained_context = True
        if retained_context:
            replaced_context = True
            LOG.info("H3 Auto Director: 二采复用已存在的上一段视频上下文 keyframe，并完成尺寸适配")

    # If no context payload survived the conditioning path, inject only an
    # explicitly supplied predecessor latent.  Never synthesize context from
    # the current segment's first-pass latent.
    if not replaced_context and prepared and torch.is_tensor(source_video) and source_video.ndim == 5:
        run = _h3_context_run(FRAME_CONTEXT_DEFAULT)
        # Keep the same temporal context length as stage one. The fallback
        # path has no existing keyframe to edit, so it injects the full tail.
        steps = {1: 1, 5: 2, 22: 7, 39: 12}[run]
        source_video_resized = _resize_h3_context_latent(
            source_video, target_video.shape[-2], target_video.shape[-1]
        )
        if int(source_video_resized.shape[2]) >= steps and steps < _h3_pixel_frames(int(target_video.shape[2])):
            entry = prepared[0]
            values = entry[1].copy()
            tail = source_video_resized[:, :, -steps:].to(
                device=target_video.device, dtype=target_video.dtype
            )
            if _native_h3_add_guide_supported():
                keyframes = list(values.get("minimax_keyframes") or [])
                keyframes.append({"resolved_frame_index": 0, "latent": tail,
                                  _NATIVE_CONTEXT_KEY: True})
                if torch.is_tensor(source_audio) and source_audio.ndim == 4:
                    audio_tail, _audio_steps, _overhang = _h3_audio_tail_from_latent(
                        {"samples": [source_video, source_audio]}, run
                    )
                    keyframes[-1]["audio_latent"] = audio_tail.to(
                        device=target_video.device, dtype=target_video.dtype
                    )
                values["minimax_keyframes"] = keyframes
                values["visual_cond_noise_aug"] = 1.0
            else:
                from . import legacy_h3_motion
                if legacy_h3_motion.ensure_legacy_h3_motion_context():
                    values.setdefault("minimax_refs", []).append({
                        "kind": "video", "latent": tail, "latent_t": int(steps),
                        "latent_h": int(target_video.shape[3]), "latent_w": int(target_video.shape[4]),
                        "ref_audio_t": 0,
                        legacy_h3_motion.MC_KEY: 0.0,
                    })
            values.pop(_MOTION_CONTEXT_MARKER, None)
            values.pop("h3_stage2_context_latent", None)
            prepared[0] = [entry[0], values, *entry[2:]]
            LOG.warning(
                "H3 Auto Director: 二采上下文标记未保留，已将上一段最终二采尾部重新注入（%d latent steps）",
                int(steps),
            )
    return adapt_keyframe_grids(prepared)


def _attach_union_control_conditioning(conditioning, control_config):
    """Attach persisted Union control metadata without copying control tensors.

    The native H3 transformer ignores unknown conditioning values, while a
    VideoX-Fun Union adapter can consume these paths and scales. Keeping only
    paths here avoids duplicating every pose/depth frame through the sampler.
    """
    if not isinstance(control_config, dict):
        return conditioning
    if not bool(control_config.get("enabled")):
        return conditioning
    payload = {
        "type": "h3_union_control",
        "control_mode": str(control_config.get("control_mode", "姿态+深度")),
        "pose_weight": float(control_config.get("pose_weight", 0.0) or 0.0),
        "depth_weight": float(control_config.get("depth_weight", 0.0) or 0.0),
        "segment_index": int(control_config.get("segment_index", 1) or 1),
        "fps": float(control_config.get("fps", FPS) or FPS),
        "frame_grid": str(control_config.get("frame_grid", "17*n+5")),
        "segments": copy.deepcopy(control_config.get("segments", {})),
        "requires_videox_fun": bool(control_config.get("requires_videox_fun", True)),
    }
    LOG.info(
        "H3 Auto Director: 已将 Union 控制接入采样条件：模式=%s，姿态权重=%.3f，深度权重=%.3f，片段=%d",
        payload["control_mode"], payload["pose_weight"], payload["depth_weight"], payload["segment_index"],
    )
    return node_helpers.conditioning_set_values(conditioning, {"h3_union_control": payload})


def _flatten_video_frames(images):
    """Flatten video frames for image upscalers and preserve the input layout.

    ComfyUI's IMAGE convention is ``[B,T,H,W,C]`` for video, but a few
    third-party VAE wrappers expose ``[B,C,T,H,W]`` directly.  Treating the
    latter as channels-last silently makes the channel axis the frame axis
    and causes a temporal mismatch when the result is encoded again.
    """
    if not torch.is_tensor(images) or images.ndim not in (4, 5):
        actual = getattr(images, "ndim", "non-tensor")
        raise ValueError(f"视频放大需要 4 或 5 维 IMAGE 张量，实际为 {actual} 维")
    if images.ndim == 4:
        # IMAGE batches are normally [B,H,W,C].  Accept [B,C,H,W] too.
        if images.shape[-1] not in (1, 3, 4) and images.shape[1] in (1, 3, 4):
            images = images.movedim(1, -1)
        return images, lambda result: result
    # Normalize to channels-last before flattening.  The channel dimension is
    # unambiguous for video tensors because it is one of 1/3/4 channels.
    channels_first = images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4)
    if channels_first:
        normalized = images.permute(0, 2, 3, 4, 1)
    else:
        normalized = images
    batches, frames, height, width, channels = normalized.shape
    flattened = normalized.reshape(batches * frames, height, width, channels)

    def restore(result):
        restored = result.reshape(batches, frames, result.shape[1], result.shape[2], result.shape[3])
        # Always return ComfyUI's IMAGE layout.  VAE.encode expects channels
        # last even when a third-party decoder returned channels first.
        return restored

    return flattened, restore


def _expected_decoded_frames(video_vae, latent):
    """Return the canonical pixel-frame count represented by an H3 latent.

    H3's VAE decoder can expose a shorter chunked output shape in some
    ComfyUI builds (for example, ``T_lat=37`` reported as 107 frames).  The
    H3 sampler however expects its canonical 17k+5 duration grid: that same
    latent represents 124 frames.  Use the model's invariant first, otherwise
    second-stage encoding drops one 17-frame chunk and returns ``T_lat=32``.
    """
    # H3's temporal compression is 17 input frames -> 5 latent frames, with
    # five leading frames represented by two latent frames.
    if torch.is_tensor(latent) and latent.ndim >= 3 and latent.shape[1] == 24:
        latent_t = int(latent.shape[2])
        if latent_t == 2:
            return 5
        if latent_t > 2 and (latent_t - 2) % 5 == 0:
            return ((latent_t - 2) // 5) * 17 + 5
    # Only non-H3/custom latent formats should rely on the VAE's wrapper
    # metadata.  This maintains graceful compatibility without compromising
    # H3's fixed temporal mapping above.
    shape_fn = getattr(getattr(video_vae, "first_stage_model", None), "decode_output_shape", None)
    if callable(shape_fn):
        try:
            shape = tuple(int(x) for x in shape_fn(tuple(latent.shape)))
            if len(shape) == 5:
                return shape[2]
        except Exception:
            # Custom VAEs do not have to expose decode_output_shape.
            pass
    return None


def _match_video_frame_count(images, target_frames):
    """Pad/crop only the temporal tail so a VAE round-trip keeps H3's length."""
    if target_frames is None or not torch.is_tensor(images):
        return images
    if images.ndim == 4:
        # H3's public IMAGE video representation is [frames,H,W,C].
        current = int(images.shape[0])
        if target_frames == current:
            return images
        if target_frames <= 0:
            raise ValueError("放大后的目标帧数必须大于 0")
        if current > target_frames:
            return images[:int(target_frames)]
        if current < 1:
            raise ValueError("放大后视频没有可用于补齐的帧")
        return torch.cat((images, images[-1:].repeat(int(target_frames) - current, 1, 1, 1)), dim=0)
    if images.ndim != 5:
        return images
    channels_first = images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4)
    time_dim = 2 if channels_first else 1
    current = int(images.shape[time_dim])
    target_frames = int(target_frames)
    if current == target_frames:
        return images
    if current > target_frames:
        index = [slice(None)] * 5
        index[time_dim] = slice(0, target_frames)
        return images[tuple(index)]
    pad = target_frames - current
    index = [slice(None)] * 5
    index[time_dim] = slice(current - 1, current)
    tail = images[tuple(index)].repeat(*(1 if i != time_dim else pad for i in range(5)))
    return torch.cat((images, tail), dim=time_dim)


def _decode_h3_video(video_vae, latent):
    """Decode H3 video and restore its canonical 17k+5 frame count.

    Most current H3 VAE builds already return the canonical length.  The
    normalization also handles wrappers that expose a shorter final temporal
    chunk, so preview, dual sampling, and final saving always agree on the
    same 24-fps timeline.
    """
    images = video_vae.decode(latent)
    # ComfyUI's VAE wrapper returns [B,T,H,W,C] for video while all H3
    # reference, Motion Context, and ffmpeg helpers consume [T,H,W,C].
    # H3 generation runs with B=1; retaining that dimension makes ffmpeg
    # treat the whole video as a single malformed frame.
    if torch.is_tensor(images) and images.ndim == 5:
        if images.shape[0] != 1:
            raise ValueError("H3 视频解码只支持批次大小为 1，实际形状：%s" % (tuple(images.shape),))
        images = images[0]
    return _match_video_frame_count(images, _expected_decoded_frames(video_vae, latent))


def _encode_h3_video(video_vae, images):
    """Encode an aligned H3 video without generic VAE temporal cropping.

    ComfyUI's generic ``VAE.encode`` crop helper uses the spatial 16x ratio
    for every IMAGE dimension. For a valid H3 124-frame video it therefore
    silently crops the time axis to 112 frames before encoding, producing
    ``T=32`` rather than ``T=37``. H3 owns temporal alignment itself, while
    the caller already guarantees 32-pixel spatial alignment, so disable only
    that generic crop for this single encode call.
    """
    if not torch.is_tensor(images) or images.ndim not in (4, 5):
        return video_vae.encode(images)
    had_crop_setting = hasattr(video_vae, "crop_input")
    original_crop_setting = getattr(video_vae, "crop_input", None)
    if had_crop_setting:
        video_vae.crop_input = False
    try:
        return video_vae.encode(images)
    finally:
        if had_crop_setting:
            video_vae.crop_input = original_crop_setting


def _match_latent_time(latent, target_time):
    """Keep H3 latent T stable after a VAE round-trip, using tail replication."""
    if not torch.is_tensor(latent) or latent.ndim < 3 or target_time is None:
        return latent
    target_time = int(target_time)
    current = int(latent.shape[2])
    if current == target_time:
        return latent
    delta = abs(current - target_time)
    if delta > 1:
        raise ValueError(
            "放大后视频 VAE latent 的时间维无法对齐："
            f"当前={tuple(latent.shape)}，目标时间长度={target_time}"
        )
    if current > target_time:
        return latent[:, :, :target_time, ...]
    tail = latent[:, :, -1:, ...].repeat(1, 1, target_time - current, *([1] * (latent.ndim - 3)))
    return torch.cat((latent, tail), dim=2)


def _upscale_interpolate(images, width, height):
    """Upscale every video frame while preserving the original batch/frame layout."""
    frames, restore = _flatten_video_frames(images)
    result = comfy.utils.common_upscale(
        frames.movedim(-1, 1), int(width), int(height), "bicubic", "disabled"
    ).movedim(1, -1)
    return restore(result)


def _upscale_with_model(upscale_model, images):
    """Use ComfyUI's tiled upscale implementation without adding graph nodes."""
    if upscale_model is None:
        raise ValueError("已选择普通放大模型，但未连接放大模型")
    patcher = getattr(upscale_model, "patcher", None)
    model = getattr(upscale_model, "model", None)
    scale = float(getattr(upscale_model, "scale", 1.0))
    if patcher is None or model is None:
        raise TypeError("普通放大模型输入不是有效的 UPSCALE_MODEL")
    frames, restore = _flatten_video_frames(images)
    memory_required = (512 * 512 * 3) * frames.element_size() * max(scale, 1.0) * 384.0
    memory_required += frames.nelement() * frames.element_size()
    model_management.load_models_gpu([patcher], memory_required=memory_required, force_full_load=True)
    in_img = frames.movedim(-1, -3).to(patcher.load_device)
    tile, overlap = 512, 32
    while True:
        try:
            steps = frames.shape[0] * comfy.utils.get_tiled_scale_steps(
                in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
            )
            pbar = comfy.utils.ProgressBar(steps)
            output = comfy.utils.tiled_scale(
                in_img, lambda part: model(part.float()), tile_x=tile, tile_y=tile,
                overlap=overlap, upscale_amount=scale, pbar=pbar,
                output_device=model_management.intermediate_device(),
            )
            result = torch.clamp(output.movedim(-3, -1), 0.0, 1.0).to(model_management.intermediate_dtype())
            return restore(result)
        except Exception as exc:
            model_management.raise_non_oom(exc)
            tile //= 2
            if tile < 128:
                raise


def _upscale_rtx(images, width, height, quality="HIGH"):
    """Run NVIDIA RTX Video Super Resolution when nvvfx is installed."""
    try:
        import nvvfx
    except ImportError as exc:
        raise RuntimeError("当前环境未安装 NVIDIA nvvfx，不能使用 RTX Video Super Resolution") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("RTX Video Super Resolution 需要可用的 NVIDIA CUDA 设备")
    quality_map = {
        "低": nvvfx.effects.QualityLevel.LOW,
        "中": nvvfx.effects.QualityLevel.MEDIUM,
        "高": nvvfx.effects.QualityLevel.HIGH,
        "最高": nvvfx.effects.QualityLevel.ULTRA,
    }
    frames, restore = _flatten_video_frames(images)
    output_width = max(32, round(int(width) / 32) * 32)
    output_height = max(32, round(int(height) / 32) * 32)
    gpu_index = int(torch.cuda.current_device())
    with nvvfx.VideoSuperRes(quality_map.get(str(quality), nvvfx.effects.QualityLevel.HIGH), device=gpu_index) as sr:
        sr.output_width, sr.output_height = output_width, output_height
        sr.load()
        LOG.info("H3 Auto Director: RTX VSR GPU=%s，输入=%s，输出=%sx%s，质量=%s",
                 torch.cuda.get_device_name(gpu_index),
                 tuple(frames.shape), output_width, output_height, quality)
        out = torch.empty((frames.shape[0], output_height, output_width, frames.shape[-1]),
                          device=frames.device, dtype=frames.dtype)
        for index in range(frames.shape[0]):
            frame = frames[index].to(device="cuda").permute(2, 0, 1).float().contiguous()
            result = torch.from_dlpack(sr.run(frame).image)
            if result.ndim != 3:
                raise RuntimeError(f"RTX Video Super Resolution 返回了异常帧维度: {tuple(result.shape)}")
            if result.shape[0] == frames.shape[-1]:
                result = result.movedim(0, -1)
            elif result.shape[-1] != frames.shape[-1]:
                raise RuntimeError(f"RTX Video Super Resolution 返回了异常通道数: {tuple(result.shape)}")
            out[index] = result.to(device=out.device, dtype=out.dtype)
            del frame, result
        # nvvfx runs asynchronously on the selected CUDA device.  Synchronize
        # before leaving the effect scope so the next chunk cannot overlap the
        # previous one and make VRAM appear to grow monotonically.
        torch.cuda.synchronize()
    return restore(out)


class H3AutoDirectorVideoSRVFI:
    """Streaming video super-resolution + frame interpolation.

    VFI is delegated to the installed ComfyUI-Frame-Interpolation nodes. Source
    frames are super-resolved first, then sent to VFI in small overlapping
    windows. Only one VFI window and one RTX-VSR frame are resident on the GPU
    at a time; output frames are moved back to CPU before the next window.
    """

    MODEL_CHOICES = (
        "AMT-G（通用高质量）",
        "GMFSS Fortuna（动漫优先）",
        "RIFE 4.9（速度优先）",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "input_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            "interpolation_multiplier": ("INT", {"default": 2, "min": 2, "max": 8}),
            "vfi_model": (list(cls.MODEL_CHOICES), {"default": cls.MODEL_CHOICES[0]}),
            "sr_frame_count": ("INT", {"default": 8, "min": 2, "max": 64,
                                      "tooltip": "每批一次送入超分模型处理的帧数；越小越省显存。"}),
            "sr_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                            "tooltip": "相对原视频分辨率的倍率；目标宽高会按源视频尺寸计算并对齐到 32 的倍数。"}),
            "sr_quality": (["低", "中", "高", "最高"], {"default": "最高"}),
        }, "optional": {
            "frames": ("IMAGE",),
            "video_source": ("H3_VIDEO_SOURCE",),
            "enable_rtx_vsr": ("BOOLEAN", {"default": True, "tooltip": "开启后使用 NVIDIA RTX Video Super Resolution；失败时明确报错，不会静默产生低质量结果。"}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("超分补帧视频", "输出帧率", "处理信息", "超分处理帧数")
    FUNCTION = "process"
    CATEGORY = "H3 自动导演/视频工具"

    @staticmethod
    def _run_vfi(model_name, frames, multiplier):
        mapping = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
        if model_name.startswith("AMT"):
            cls = mapping.get("AMT VFI")
            kwargs = {"ckpt_name": "amt-g.pth"}
        elif model_name.startswith("GMFSS"):
            cls = mapping.get("GMFSS Fortuna VFI")
            kwargs = {"ckpt_name": "GMFSS_fortuna_union"}
        else:
            cls = mapping.get("RIFE VFI")
            kwargs = {"ckpt_name": "rife49.pth", "fast_mode": True, "ensemble": False, "scale_factor": 1.0}
        if cls is None:
            raise RuntimeError("未找到视频补帧节点，请安装并启用 ComfyUI-Frame-Interpolation（AMT/GMFSS/RIFE）")
        kwargs.update({"frames": frames, "clear_cache_after_n_frames": 1, "multiplier": int(multiplier)})
        # VFI is an inference-only stage.  Explicitly disable autograd so a
        # third-party interpolation node cannot retain computation graphs for
        # every processed window.
        with torch.inference_mode():
            result = cls().vfi(**kwargs)
        if not isinstance(result, (tuple, list)) or not result or not torch.is_tensor(result[0]):
            raise RuntimeError("视频补帧节点返回了无效的 IMAGE")
        return result[0].detach().cpu().contiguous()

    def process(self, input_fps, interpolation_multiplier, vfi_model,
                sr_frame_count, sr_scale, sr_quality, frames=None,
                video_source=None, enable_rtx_vsr=True):
        if video_source is not None:
            if not isinstance(video_source, dict) or not video_source.get("path"):
                raise ValueError("视频源句柄无效")
            frames = H3AutoDirectorVideoLoad.decode_all(video_source.get("path"), int(sr_frame_count))
        if not torch.is_tensor(frames) or frames.ndim not in (4, 5):
            raise ValueError("视频超分补帧输入必须是 4/5 维 IMAGE")
        flat, _restore = _flatten_video_frames(frames)
        # Keep only the flattened view; retaining both names needlessly keeps
        # an additional Python reference alive throughout the long operation.
        del frames
        if flat.shape[0] < 2:
            raise ValueError("视频补帧至少需要 2 帧")
        try:
            multiplier_value = float(interpolation_multiplier)
            multiplier = max(2, int(multiplier_value)) if math.isfinite(multiplier_value) else 2
        except (TypeError, ValueError, OverflowError):
            multiplier = 2
        try:
            source_fps_value = float(input_fps)
            source_fps = max(1.0, source_fps_value) if math.isfinite(source_fps_value) else 24.0
        except (TypeError, ValueError, OverflowError):
            source_fps = 24.0
        # VFI requires a pair of adjacent source frames.  A one-frame chunk
        # cannot produce output until the next chunk arrives, so normalize the
        # lower bound to two and avoid dropping the first frame at the stream
        # boundary.
        sr_frame_count = min(64, max(2, int(sr_frame_count)))
        try:
            scale = float(sr_scale)
        except (TypeError, ValueError):
            scale = 2.0
        scale = min(4.0, max(1.0, scale)) if math.isfinite(scale) else 2.0
        source_height, source_width = int(flat.shape[1]), int(flat.shape[2])
        target_width = max(32, round(source_width * scale / 32) * 32)
        target_height = max(32, round(source_height * scale / 32) * 32)
        # Super-resolve source frames before VFI.  Keep this operation in small
        # batches as a memory guard.  The VFI window below remains fixed and
        # is intentionally not exposed as a user setting.
        vfi_window_size = 8
        # Process one super-resolution chunk directly into VFI windows.  The
        # previous implementation accumulated every upscaled chunk in
        # ``sr_frames``, then duplicated it with ``torch.cat`` and accumulated
        # a second full copy in ``outputs``.  Long videos therefore consumed
        # nearly all system RAM even though GPU tensors were released.  Keep
        # only the source tensor (owned by the upstream graph), one chunk and
        # one preallocated final output buffer.
        source_total = int(flat.shape[0])
        expected_output = max(1, (source_total - 1) * multiplier + 1)
        result = None
        written = 0
        previous_tail = None
        for chunk_start in range(0, source_total, sr_frame_count):
            source_window = flat[chunk_start:min(source_total, chunk_start + sr_frame_count)].detach().cpu().contiguous()
            if bool(enable_rtx_vsr):
                upscaled = _upscale_rtx(source_window, target_width, target_height, str(sr_quality))
            else:
                upscaled = _upscale_interpolate(source_window, target_width, target_height)
            del source_window
            upscaled = upscaled.detach().cpu().contiguous()
            # Share one source frame across super-resolution chunks.  The
            # first generated frame of every chunk after the first is the
            # shared boundary and is discarded below.
            boundary_chunk = previous_tail is not None
            if previous_tail is not None:
                vfi_input = torch.cat((previous_tail, upscaled), dim=0)
            else:
                vfi_input = upscaled
            previous_tail = upscaled[-1:].clone()
            window_start = 0
            input_total = int(vfi_input.shape[0])
            while window_start < input_total - 1:
                window_end = min(input_total, window_start + vfi_window_size)
                window = vfi_input[window_start:window_end].detach().cpu().contiguous()
                interpolated = self._run_vfi(vfi_model, window, multiplier).detach().cpu().contiguous()
                drop_first = boundary_chunk or window_start > 0
                if drop_first and interpolated.shape[0] > 0:
                    interpolated = interpolated[1:]
                if interpolated.shape[0] > 0:
                    if result is None:
                        result = torch.empty((expected_output, *interpolated.shape[1:]),
                                             dtype=interpolated.dtype, device="cpu")
                    count = min(int(interpolated.shape[0]), int(result.shape[0]) - written)
                    if count > 0:
                        result[written:written + count].copy_(interpolated[:count])
                        written += count
                del window, interpolated
                window_start = window_end - 1
                if window_end >= input_total:
                    break
            del vfi_input, upscaled
            _release_video_memory()
        if result is None or written < 1:
            raise RuntimeError("超分补帧未产生有效输出帧")
        # This is a view into the preallocated buffer; making it contiguous
        # here would duplicate the entire output one more time.
        result = result[:written]
        total = source_total
        out_fps = source_fps * multiplier
        info = (f"模型={vfi_model}，输入={total}帧，输出={int(result.shape[0])}帧，"
                f"源帧率={source_fps:.3f}，超分批大小={sr_frame_count}帧，输出帧率={out_fps:.3f}，"
                f"原视频={source_width}x{source_height}，超分倍率={scale:.2f}x，"
                f"输出={target_width}x{target_height}，VFI窗口=8，RTX VSR={'开启' if enable_rtx_vsr else '关闭'}")
        LOG.info("H3 Auto Director: %s", info)
        return (result, out_fps, info, int(sr_frame_count))


class H3AutoDirectorVideoLoad:
    """Load a video incrementally from disk into an IMAGE sequence.

    Frames are decoded in small CPU batches and immediately released after
    conversion.  This keeps decoder peak memory bounded and provides source
    FPS/size metadata to the following nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # The path is an internal serialized value populated by the upload
        # button.  It is hidden so the node exposes only a direct file upload
        # control instead of a misleading directory/path text box.
        return {"required": {
            "video_path": ("STRING", {"default": "", "multiline": False, "hidden": True}),
        }}

    RETURN_TYPES = ("H3_VIDEO_SOURCE", "FLOAT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("视频源", "原视频帧率", "宽度", "高度", "总帧数", "视频信息")
    FUNCTION = "load"
    CATEGORY = "H3 自动导演/视频工具"

    @staticmethod
    def decode_all(video_path, decode_batch=8):
        source = _direct_video_file(video_path)
        container = av.open(str(source), "r")
        chunks, window = [], []
        try:
            streams = tuple(container.streams.video)
            if not streams:
                raise ValueError("输入文件不包含视频流")
            stream = streams[0]
            width = int(getattr(stream, "width", 0) or getattr(stream.codec_context, "width", 0) or 0)
            height = int(getattr(stream, "height", 0) or getattr(stream.codec_context, "height", 0) or 0)
            expected = int(getattr(stream, "frames", 0) or 0)
            # PyAV usually exposes an exact frame count for regular video
            # files.  Preallocate once in that case instead of retaining a
            # list of chunks and duplicating the whole source in torch.cat.
            preallocated = torch.empty((expected, height, width, 3), dtype=torch.float32) if expected > 0 and width > 0 and height > 0 else None
            decoded = 0
            for frame in container.decode(stream):
                image = torch.from_numpy(frame.to_ndarray(format="rgb24")).float().div_(255.0)
                if preallocated is not None and decoded < preallocated.shape[0]:
                    preallocated[decoded].copy_(image)
                else:
                    window.append(image)
                    if len(window) >= max(1, int(decode_batch)):
                        chunks.append(torch.stack(window, dim=0)); window.clear()
                decoded += 1
            if preallocated is not None and decoded > 0:
                if decoded <= preallocated.shape[0]:
                    return preallocated[:decoded].contiguous()
                # A malformed stream reported too few frames; retain the
                # overflow frames through the normal chunk fallback.
                chunks.insert(0, preallocated)
            if window: chunks.append(torch.stack(window, dim=0))
            if not chunks: raise ValueError("输入视频没有可解码帧")
            return torch.cat(chunks, dim=0).contiguous()
        finally:
            container.close(); gc.collect()

    def load(self, video_path):
        if av is None:
            raise RuntimeError("加载视频需要 PyAV；请安装 av 后重启 ComfyUI")
        source = _direct_video_file(video_path)
        try:
            container = av.open(str(source), "r")
        except Exception as exc:
            raise RuntimeError("无法读取输入视频：%s" % source) from exc
        try:
            streams = tuple(container.streams.video)
            if not streams:
                raise ValueError("输入文件不包含视频流")
            stream = streams[0]
            rate = stream.average_rate or stream.base_rate
            fps = float(rate) if rate is not None and float(rate) > 0 else 24.0
            width = int(getattr(stream, "width", 0) or getattr(stream.codec_context, "width", 0) or 0)
            height = int(getattr(stream, "height", 0) or getattr(stream.codec_context, "height", 0) or 0)
            total = int(getattr(stream, "frames", 0) or 0)
            info = f"{source} | {width}x{height} | {fps:.3f} FPS | {total or '未知'} 帧"
            LOG.info("H3 Auto Director: 加载视频 %s", info)
            return ({"path": str(source), "fps": fps, "width": width, "height": height, "frames": total}, fps, width, height, total, info)
        finally:
            container.close()
            gc.collect()


class H3AutoDirectorVideoSave:
    """Stream an IMAGE sequence to ffmpeg without making an extra copy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            "filename": ("STRING", {"default": "H3_video.mp4", "multiline": False}),
        }, "optional": {
            "sr_frame_count": ("INT", {"default": 8, "min": 1, "max": 64,
                                             "tooltip": "由超分补帧节点输出的超分处理帧数；未连接时使用 8。"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("已保存视频", "保存信息")
    FUNCTION = "save"
    CATEGORY = "H3 自动导演/视频工具"
    OUTPUT_NODE = True

    def save(self, frames, fps=24.0, filename="H3_video.mp4", sr_frame_count=8):
        if not torch.is_tensor(frames) or frames.ndim not in (4, 5):
            raise ValueError("保存视频输入必须是 4/5 维 IMAGE")
        flat, _ = _flatten_video_frames(frames)
        if flat.shape[0] < 1:
            raise ValueError("没有可保存的视频帧")
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("未找到 ffmpeg，无法保存视频")
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(filename or "H3_video.mp4").strip()).strip("._") or "H3_video.mp4"
        if not Path(clean).suffix:
            clean += ".mp4"
        output = _output_root() / "h3_video_tools" / clean
        output.parent.mkdir(parents=True, exist_ok=True)
        height, width = int(flat.shape[1]), int(flat.shape[2])
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{float(fps):.8f}",
                   "-i", "pipe:0", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        chunk = max(1, int(sr_frame_count))
        try:
            for start in range(0, int(flat.shape[0]), chunk):
                data = flat[start:start + chunk].detach().cpu().clamp(0, 1).mul(255).round().byte().numpy()[..., :3]
                process.stdin.write(data.tobytes())
                del data
            process.stdin.close()
            stderr = process.stderr.read().decode(errors="replace")
            if process.wait(timeout=600) != 0:
                raise RuntimeError(stderr[-2000:] or "ffmpeg 编码失败")
        except Exception:
            if process.poll() is None:
                process.terminate()
            output.unlink(missing_ok=True)
            raise
        info = f"{output} | {int(flat.shape[0])} 帧 | {float(fps):.3f} FPS | 编码块={chunk}"
        LOG.info("H3 Auto Director: 保存视频 %s", info)
        return (str(output), info)


class H3AutoDirectorStreamingVideoSRVFI:
    """Disk-to-disk RTX VSR and VFI processing for long videos.

    Unlike an IMAGE based graph, this node never creates a tensor containing
    the full source or output video.  PyAV decodes a small overlapping source
    window, the VFI model returns that window, RTX VSR processes it one frame
    at a time, and ffmpeg immediately encodes the result to disk.
    """

    MODEL_CHOICES = H3AutoDirectorVideoSRVFI.MODEL_CHOICES

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path": ("STRING", {"default": "", "multiline": False, "hidden": True}),
            "interpolation_multiplier": ("INT", {"default": 2, "min": 2, "max": 8}),
            "vfi_model": (list(cls.MODEL_CHOICES), {"default": cls.MODEL_CHOICES[0]}),
            "chunk_size": ("INT", {"default": 8, "min": 2, "max": 64,
                            "tooltip": "单次解码/补帧的源帧数；4 更省显存，8 是推荐默认值。"}),
            "sr_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                            "tooltip": "相对原视频分辨率的倍率；目标宽高按源视频尺寸计算并对齐到 32 的倍数。"}),
            "sr_quality": (["低", "中", "高", "最高"], {"default": "最高"}),
            "filename_prefix": ("STRING", {"default": "H3_Video_SR_VFI"}),
            "preserve_audio": ("BOOLEAN", {"default": True,
                                "tooltip": "从源视频读取并重新封装音轨；不会把音频载入 PyTorch。"}),
        }, "optional": {
            "enable_rtx_vsr": ("BOOLEAN", {"default": True,
                               "tooltip": "开启后必须使用 NVIDIA RTX Video Super Resolution；未安装 nvvfx 时会明确报错。"}),
        }}

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("已保存视频", "输出帧率", "处理信息")
    FUNCTION = "process"
    CATEGORY = "H3 自动导演/视频工具"
    OUTPUT_NODE = True

    @staticmethod
    def _output_path(prefix):
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(prefix or "H3_Video_SR_VFI").strip()).strip("._")
        clean = clean or "H3_Video_SR_VFI"
        directory = _output_root() / "h3_video_tools"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{clean}_{uuid.uuid4().hex[:8]}.mp4"

    @staticmethod
    def _encoder_command(ffmpeg, output, source, width, height, fps, preserve_audio):
        # The output duration is unchanged by interpolation: only its frame
        # rate rises.  Re-encode optional source audio so uncommon input audio
        # codecs cannot make an otherwise successful video encode fail.
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                   "-r", f"{fps:.8f}", "-i", "pipe:0"]
        if preserve_audio:
            command.extend(["-i", str(source), "-map", "0:v:0", "-map", "1:a?"])
        else:
            command.extend(["-map", "0:v:0", "-an"])
        command.extend(["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"])
        if preserve_audio:
            command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        command.extend(["-movflags", "+faststart", "-f", "mp4", str(output)])
        return command

    @staticmethod
    def _write_chunk(process, images):
        if images.numel() == 0:
            return 0
        rgb = images.detach().cpu().clamp(0, 1).mul(255).round().byte().numpy()[..., :3]
        try:
            process.stdin.write(rgb.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("ffmpeg 视频编码进程提前结束") from exc
        return int(rgb.shape[0])

    def process(self, video_path, interpolation_multiplier, vfi_model, chunk_size,
                sr_scale, sr_quality, filename_prefix, preserve_audio,
                enable_rtx_vsr=True):
        if av is None:
            raise RuntimeError("流式视频超分补帧需要 PyAV；请安装 av 后重启 ComfyUI")
        source = _direct_video_file(video_path)
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("未找到 ffmpeg，无法流式写入超分补帧视频。请设置 FFMPEG_PATH 或安装 imageio-ffmpeg。")
        chunk_size = max(2, int(chunk_size))
        multiplier = max(2, int(interpolation_multiplier))
        output = self._output_path(filename_prefix)

        try:
            container = av.open(str(source), "r")
        except Exception as exc:
            raise RuntimeError("无法读取输入视频：%s" % exc) from exc
        encode_process = None
        source_frames = 0
        output_frames = 0
        try:
            streams = tuple(container.streams.video)
            if not streams:
                raise RuntimeError("输入文件不包含视频流")
            stream = streams[0]
            source_width = int(getattr(stream, "width", 0) or getattr(stream.codec_context, "width", 0) or 0)
            source_height = int(getattr(stream, "height", 0) or getattr(stream.codec_context, "height", 0) or 0)
            if source_width < 1 or source_height < 1:
                raise RuntimeError("无法读取输入视频分辨率")
            try:
                scale = float(sr_scale)
            except (TypeError, ValueError):
                scale = 2.0
            scale = min(4.0, max(1.0, scale))
            target_width = max(32, round(source_width * scale / 32) * 32)
            target_height = max(32, round(source_height * scale / 32) * 32)
            rate = stream.average_rate or stream.base_rate
            if rate is None or float(rate) <= 0:
                raise RuntimeError("无法读取输入视频帧率")
            input_fps = float(rate)
            output_fps = input_fps * multiplier
            encode_process = subprocess.Popen(
                self._encoder_command(ffmpeg, output, source, target_width, target_height,
                                      output_fps, bool(preserve_audio)),
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            window = []
            has_previous_window = False
            for frame in container.decode(stream):
                # The decoded RGB frame lives only until its small overlapping
                # VFI window has been written to ffmpeg.
                window.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).float().div_(255.0))
                source_frames += 1
                if len(window) < chunk_size:
                    continue
                interpolated = H3AutoDirectorVideoSRVFI._run_vfi(
                    vfi_model, torch.stack(window, dim=0), multiplier)
                if bool(enable_rtx_vsr):
                    interpolated = _upscale_rtx(interpolated, target_width, target_height, str(sr_quality))
                else:
                    interpolated = _upscale_interpolate(interpolated, target_width, target_height)
                if has_previous_window:
                    interpolated = interpolated[1:]
                output_frames += self._write_chunk(encode_process, interpolated)
                # Adjacent windows share exactly one source frame, preserving
                # all VFI intervals while avoiding a duplicate output frame.
                window = [window[-1]]
                has_previous_window = True
                del interpolated
                _release_video_memory()
            if len(window) >= 2:
                interpolated = H3AutoDirectorVideoSRVFI._run_vfi(
                    vfi_model, torch.stack(window, dim=0), multiplier)
                if bool(enable_rtx_vsr):
                    interpolated = _upscale_rtx(interpolated, target_width, target_height, str(sr_quality))
                else:
                    interpolated = _upscale_interpolate(interpolated, target_width, target_height)
                if has_previous_window:
                    interpolated = interpolated[1:]
                output_frames += self._write_chunk(encode_process, interpolated)
                del interpolated
                _release_video_memory()
            if source_frames < 2:
                raise ValueError("视频补帧至少需要 2 帧")
            encode_process.stdin.close()
            stderr = encode_process.stderr.read().decode(errors="replace")
            if encode_process.wait(timeout=600) != 0:
                raise RuntimeError(stderr[-2000:] or "ffmpeg 编码失败")
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("ffmpeg 没有写出有效视频文件")
            info = (f"模型={vfi_model}，输入={source_frames}帧，输出={output_frames}帧，"
                    f"帧率={output_fps:.3f}，原视频={source_width}x{source_height}，超分倍率={scale:.2f}x，"
                    f"输出={target_width}x{target_height}，分块={chunk_size}，RTX VSR={'开启' if enable_rtx_vsr else '关闭'}，"
                    f"文件={output}")
            LOG.info("H3 Auto Director: %s", info)
            return (str(output), output_fps, info)
        except Exception:
            if encode_process is not None and encode_process.poll() is None:
                try:
                    encode_process.stdin.close()
                except Exception:
                    pass
                encode_process.terminate()
                try:
                    encode_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    encode_process.kill()
            output.unlink(missing_ok=True)
            raise
        finally:
            container.close()
            _release_video_memory()


class H3AutoDirectorResolution:
    """Calculate aligned H3 resolutions for both stages of dual sampling."""

    ASPECTS = {
        "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3),
        "3:4": (3, 4), "3:2": (3, 2), "2:3": (2, 3), "21:9": (21, 9),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "use_preset_ratio": ("BOOLEAN", {"default": True, "label_on": "使用预设", "label_off": "不使用预设"}),
            "use_custom_ratio": ("BOOLEAN", {"default": False, "label_on": "使用自定义", "label_off": "不使用自定义"}),
            "aspect_preset": (list(cls.ASPECTS), {"default": "16:9"}),
            "custom_ratio": ("STRING", {"default": "16,9", "tooltip": "输入宽,高；支持英文逗号或中文逗号，例如 16,9 或 9，16"}),
            "stage1_megapixels": ("FLOAT", {"default": 0.4, "min": 0.2, "max": 5.0, "step": 0.01}),
            "stage2_megapixels": ("FLOAT", {"default": 0.98, "min": 0.2, "max": 5.0, "step": 0.01}),
            "multiple": ("INT", {"default": 32, "min": 32, "max": 128, "step": 32,
                                  "tooltip": "H3 画布必须是 32 的倍数；旧工作流中的更小值会自动规范化。"}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("第一阶段宽度", "第一阶段高度", "第二阶段宽度", "第二阶段高度", "分辨率预览")
    FUNCTION = "calculate"
    CATEGORY = "H3 自动导演/采样"

    @staticmethod
    def _aligned_resolution(ratio_width, ratio_height, megapixels, multiple):
        ratio = float(ratio_width) / float(ratio_height)
        # Match ComfyUI's ResolutionSelector convention: 1 MP is 1024 x 1024
        # target pixels before alignment, while the preview reports actual MP.
        target_pixels = float(megapixels) * 1024 * 1024
        width = math.sqrt(target_pixels * ratio)
        height = width / ratio
        # MiniMax H3's canvas and packed latent grid are defined on a 32px
        # spatial multiple.  Older versions of this node exposed 16/24px
        # choices; those values produced dimensions that were silently
        # truncated by _empty_av_latent(), making the user's resolution look
        # ineffective.  Keep the widget backwards compatible but canonicalize
        # every result to the core H3 multiple here.
        h3_multiple = int(getattr(_minimax_h3, "CANVAS_MULTIPLE", 32) or 32)
        multiple = max(h3_multiple, int(multiple))
        multiple = max(h3_multiple, (multiple // h3_multiple) * h3_multiple)
        width = max(multiple, int(round(width / multiple)) * multiple)
        height = max(multiple, int(round(height / multiple)) * multiple)
        width, height = _h3_canvas_dimensions(width, height)
        max_dimension = max(multiple, int(getattr(nodes, "MAX_RESOLUTION", 16384)) // multiple * multiple)
        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            width = max(multiple, int(round(width * scale / multiple)) * multiple)
            height = max(multiple, int(round(height * scale / multiple)) * multiple)
        return width, height

    @staticmethod
    def _parse_custom_ratio(value):
        parts = [part.strip() for part in str(value or "").replace("，", ",").split(",")]
        if len(parts) != 2:
            raise ValueError("自定义比例请按“宽,高”输入，例如 16,9 或 9，16")
        try:
            width, height = (float(part) for part in parts)
        except ValueError as exc:
            raise ValueError("自定义比例必须是两个正数，例如 16,9") from exc
        if width <= 0 or height <= 0:
            raise ValueError("自定义比例的宽和高必须大于 0")
        return width, height

    def calculate(self, use_preset_ratio, use_custom_ratio, aspect_preset, custom_ratio,
                  stage1_megapixels, stage2_megapixels, multiple):
        # The UI keeps these switches mutually exclusive. Prefer custom here
        # too, so API callers cannot get an ambiguous result when both are set.
        if bool(use_custom_ratio):
            ratio_width, ratio_height = self._parse_custom_ratio(custom_ratio)
            ratio_label = f"{ratio_width:g}:{ratio_height:g}"
        elif bool(use_preset_ratio):
            ratio_width, ratio_height = self.ASPECTS.get(str(aspect_preset), self.ASPECTS["16:9"])
            ratio_label = str(aspect_preset)
        else:
            ratio_width, ratio_height = self.ASPECTS["16:9"]
            ratio_label = "16:9（默认）"
        first_width, first_height = self._aligned_resolution(ratio_width, ratio_height, stage1_megapixels, multiple)
        second_width, second_height = self._aligned_resolution(ratio_width, ratio_height, stage2_megapixels, multiple)
        preview = (
            f"第一阶段：{first_width} x {first_height}（{first_width * first_height / 1_000_000:.2f} MP） | "
            f"第二阶段：{second_width} x {second_height}（{second_width * second_height / 1_000_000:.2f} MP） | {ratio_label}"
        )
        LOG.info(
            "H3 Auto Director: 分辨率计算 ratio=%s stage1=%dx%d stage2=%dx%d (输入 MP=%.3f/%.3f, multiple=%d)",
            ratio_label, first_width, first_height, second_width, second_height,
            float(stage1_megapixels), float(stage2_megapixels), int(multiple),
        )
        return (first_width, first_height, second_width, second_height, preview)


class H3AutoDirectorDualSampling:
    """Two-stage H3 sampling with pixel-space upscale and AV reassembly.

    Stage one establishes motion and composition at a lower resolution. Its
    video stream is decoded, upscaled, and VAE-encoded again; its audio latent
    is preserved byte-for-byte before stage two refines the joint AV latent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "stage1_model": ("MODEL", {"tooltip": "一采模型。可直接连接模型加载器或外部 LoRA/显存优化链。"}),
            "conditioning": ("CONDITIONING",), "latent": ("LATENT",),
            "video_vae": ("VAE",), "audio_vae": ("VAE",),
            "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "res_multistep"}),
            "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
            "stage1_steps": ("INT", {"default": 6, "min": 1, "max": 100}),
            "stage1_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_stage2": ("BOOLEAN", {"default": True, "label_on": "启用二采", "label_off": "关闭二采",
                               "tooltip": "关闭后仅执行第一阶段采样，跳过放大、视频 VAE 重编码和第二阶段采样。"}),
            "stage2_use_context": ("BOOLEAN", {"default": False, "label_on": "二采使用上下文（实验性）", "label_off": "二采不使用上下文",
                                    "tooltip": "实验性功能，当前效果不可用，默认应关闭。开启后会尝试将上一段最终二采视频/音频上下文适配到二采尺寸后传入。"}),
            "stage2_steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "stage2_denoise": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            "upscale_mode": (DUAL_UPSCALE_CHOICES, {"default": "普通插值"}),
            "target_width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32, "forceInput": True}),
            "target_height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32, "forceInput": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "use_stage1_audio_only": ("BOOLEAN", {"default": False,
                                      "label_on": "最终仅使用一采音频",
                                      "label_off": "使用二采音频",
                                      "tooltip": "二采继续细化画面，但最终 AV latent 使用第一阶段生成的音频。"}),
            "enable_preview": ("BOOLEAN", {"default": False, "label_on": "开启新版采样预览", "label_off": "关闭新版采样预览"}),
            "latent_upscale_model": ((_h3_latent_upscaler.available_models() if _h3_latent_upscaler else ["(未加载 H3 latent 放大器)"]), {"default": (_h3_latent_upscaler.available_models()[0] if _h3_latent_upscaler and _h3_latent_upscaler.available_models() else "")}),
            "latent_upscale_device": (["cuda", "cpu"], {"default": "cuda"}),
            "latent_upscale_precision": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
            "stage1_extend_sigmas": ("BOOLEAN", {"default": False, "label_on": "开启一采 Sigmas 扩展", "label_off": "关闭一采 Sigmas 扩展", "tooltip": "开启后按 ComfyUI ExtendIntermediateSigmas 原版算法扩展一采调度。"}),
            "stage1_extend_steps": ("INT", {"default": 2, "min": 1, "max": 100}),
            "stage1_start_at_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20000.0, "step": 0.01, "round": False}),
            "stage1_end_at_sigma": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 20000.0, "step": 0.01, "round": False}),
            "stage1_spacing": (["linear", "cosine", "sine"], {"default": "linear"}),
            "stage2_extend_sigmas": ("BOOLEAN", {"default": False, "label_on": "开启二采 Sigmas 扩展", "label_off": "关闭二采 Sigmas 扩展", "tooltip": "开启后按 ComfyUI ExtendIntermediateSigmas 原版算法扩展二采调度。"}),
            "stage2_extend_steps": ("INT", {"default": 2, "min": 1, "max": 100}),
            "stage2_start_at_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20000.0, "step": 0.01, "round": False}),
            "stage2_end_at_sigma": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 20000.0, "step": 0.01, "round": False}),
            "stage2_spacing": (["linear", "cosine", "sine"], {"default": "linear"}),
        }, "optional": {
            "plan": ("H3_AUTO_PLAN", {"tooltip": "可选。连接项目计划后，开启统一解码时会在所有片段采样完成前禁止任何视频/音频解码。"}),
            "upscale_model": ("UPSCALE_MODEL",),
            "stage2_model": ("MODEL", {"tooltip": "可选。连接外部 LoRA/显存优化后的第二阶段模型；未连接时自动复用一采模型。"}),
            "audio_sampling": ("H3_AUDIO_SAMPLING", {"tooltip": "可选。连接‘音频采样切换’的采样调度信息；它只设置 H3 音频采样方法与偏移，不会覆盖两阶段的步数、降噪或调度器。"}),
            "stage1_sigmas": ("SIGMAS", {"tooltip": "可选。一采使用的完整 Sigmas 调度；连接后优先于一采步数、降噪和调度器。"}),
            "stage2_sigmas": ("SIGMAS", {"tooltip": "可选。二采使用的完整 Sigmas 调度；连接后优先于二采步数、降噪和调度器。"}),
            "stage2_conditioning": ("CONDITIONING", {"tooltip": "可选。二采专用正向条件；未连接时复用一采条件。"}),
            "stage2_context_latent": ("LATENT", {"tooltip": "可选。二采上下文潜变量；仅在开启二采上下文时使用。"}),
            "control_config": ("H3_CONTROL_CONFIG", {"tooltip": "可选。连接统一姿态/深度控制节点；控制视频路径与独立权重会传入采样条件。"}),
        }}

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("最终 AV latent", "第一阶段 AV latent", "放大预览", "第一阶段画面")
    FUNCTION = "sample"
    CATEGORY = "H3 自动导演/采样"

    def sample(self, stage1_model, conditioning, latent, video_vae, audio_vae, sampler_name, scheduler,
               stage1_steps, stage1_denoise, stage2_steps, stage2_denoise, upscale_mode,
               target_width, target_height, enable_stage2=True, stage2_use_context=False,
               upscale_model=None, seed=0, stage2_conditioning=None, stage2_context_latent=None,
               use_stage1_audio_only=False, enable_preview=False, latent_upscale_model=None,
               latent_upscale_device="cuda", latent_upscale_precision="fp32",
               stage2_model=None, audio_sampling=None, stage1_sigmas=None, stage2_sigmas=None,
               stage1_extend_sigmas=False, stage1_extend_steps=2, stage1_start_at_sigma=-1.0,
               stage1_end_at_sigma=12.0, stage1_spacing="linear", stage2_extend_sigmas=False,
               stage2_extend_steps=2, stage2_start_at_sigma=-1.0, stage2_end_at_sigma=12.0,
               stage2_spacing="linear", control_config=None, plan=None, **_legacy_unused):
        global _LAST_STAGE1_CONTEXT
        deferred_decode = bool(isinstance(plan, dict) and plan.get("decode_after_all_segments", False))
        if stage1_model is None:
            stage1_model = _legacy_unused.get("model")
        if stage1_model is None:
            raise ValueError("一采模型未连接：请将模型加载器或外部模型补丁链连接到双采样的一采模型输入")
        input_parts = _av_latent_parts(latent)
        if input_parts is not None and torch.is_tensor(input_parts[0]) and input_parts[0].ndim >= 5:
            input_h = int(input_parts[0].shape[-2]) * 16
            input_w = int(input_parts[0].shape[-1]) * 16
            LOG.info("H3 Auto Director: 一采输入 latent 网格=%dx%d（%.3f MP）",
                     input_w, input_h, input_w * input_h / 1_000_000)
        # A direct connection from the audio switch's SIGMAS socket is an
        # alternative to its metadata socket.  It still applies the embedded
        # video/audio shifts, but must not replace the stage's full schedule
        # with the raw two-endpoint base Tensor.
        if audio_sampling is None:
            audio_sampling = (
                _audio_sampling_from_base_sigmas(stage1_sigmas)
                or _audio_sampling_from_base_sigmas(stage2_sigmas)
            )
        conditioning = _attach_union_control_conditioning(conditioning, control_config)
        stage1_conditioning = _prepare_dual_sampling_conditioning(conditioning)
        first_model = _apply_audio_sampling_config(stage1_model, audio_sampling, "一采模型")
        second_source_model = stage2_model or stage1_model
        second_model = _apply_audio_sampling_config(second_source_model, audio_sampling, "二采模型")
        # Stage models are optional by design.  This keeps ordinary graphs
        # simple and lets users put the standard ComfyUI LoRA/optimization
        # nodes exactly where each stage needs them.
        first = _dual_sample(first_model, stage1_conditioning, latent, sampler_name, scheduler,
                             stage1_steps, stage1_denoise, seed, enable_preview, stage1_sigmas,
                             stage1_extend_sigmas, stage1_extend_steps, stage1_start_at_sigma,
                             stage1_end_at_sigma, stage1_spacing)
        # SaveSegment consumes this immediately after the sampler.  Keeping it
        # here preserves compatibility with existing graphs that do not expose
        # the optional first-stage latent socket.
        _LAST_STAGE1_CONTEXT = first
        first_output = dict(first)
        first_output["_h3_stage1_context"] = first
        stage1_audio_override = None
        if bool(use_stage1_audio_only):
            # Keep an explicit decoded copy on the latent so downstream save
            # nodes cannot accidentally use a separately connected second-pass
            # AUDIO socket.  This is only decoded when the opt-in is enabled.
            stage1_parts = _av_latent_parts(first)
            if stage1_parts is None:
                raise ValueError("一采音频选项需要 H3 联合 AV latent")
            stage1_audio_latent = _normalize_h3_audio_latent(stage1_parts[1], "一采音频 latent")
            if not deferred_decode:
                stage1_waveform, stage1_sample_rate = _decode_h3_audio(audio_vae, stage1_audio_latent)
                stage1_audio_override = {
                    "waveform": stage1_waveform.detach().cpu().contiguous(),
                    "sample_rate": int(stage1_sample_rate),
                }
                # Also expose it on the first-stage return for graphs that save
                # that branch explicitly instead of using the final branch.
                first["_h3_stage1_audio"] = stage1_audio_override
                first_output["_h3_stage1_audio"] = stage1_audio_override
        if not bool(enable_stage2):
            # Preserve the output contract without paying for decode/upscale.
            # The empty IMAGE is deliberately inert; the final AV latent is
            # the first-stage result and remains compatible with AV Decode.
            empty_preview = torch.empty((0, 1, 1, 3), dtype=torch.float32)
            first_images = (torch.empty((0, 1, 1, 3), dtype=torch.float32)
                            if deferred_decode else _decode_h3_video(video_vae, _av_latent_parts(first)[0]))
            return (first_output, first, empty_preview, first_images)
        parts = _av_latent_parts(first)
        if parts is None:
            raise ValueError("双采样仅支持 MiniMax H3 联合 AV latent")
        first_video, first_audio = parts
        decoded = None if deferred_decode else _decode_h3_video(video_vae, first_video)
        width, height = _h3_canvas_dimensions(target_width, target_height)
        LOG.info("H3 Auto Director: 二采目标画布=%dx%d（%.3f MP）",
                 width, height, width * height / 1_000_000)
        target_latent_size = (max(1, height // 16), max(1, width // 16))
        first_latent_size = (int(first_video.shape[-2]), int(first_video.shape[-1]))
        if first_latent_size == target_latent_size:
            # Same spatial grid: a pixel upscale followed by VAE encode would
            # only add reconstruction loss.  Keep the first-pass video latent
            # byte-for-byte and decode it only for the preview output.
            encoded_video = first_video
            preview = (torch.empty((0, 1, 1, 3), dtype=torch.float32)
                       if deferred_decode else decoded)
            LOG.info(
                "H3 Auto Director: 一采/二采分辨率相同（latent=%s），跳过放大与视频 VAE 重编码",
                first_latent_size,
            )
        else:
            expected_frames = _expected_decoded_frames(video_vae, first_video)
            mode = str(upscale_mode)
            if mode == "H3 Latent 学习型放大":
                if _h3_latent_upscaler is None:
                    raise RuntimeError("H3 latent 放大器未加载")
                encoded_video = _h3_latent_upscaler.upscale_video(
                    first_video, latent_upscale_model,
                    target_latent_size[0], target_latent_size[1],
                    latent_upscale_device, latent_upscale_precision)
                preview = (torch.empty((0, 1, 1, 3), dtype=torch.float32)
                           if deferred_decode else _decode_h3_video(video_vae, encoded_video))
            elif mode == "RTX Video Super Resolution":
                if deferred_decode:
                    raise RuntimeError("统一解码模式下不能使用 RTX 视频放大：请改用 H3 Latent 学习型放大或在一采/二采使用相同分辨率")
                preview = _upscale_rtx(decoded, width, height, "高")
            elif mode == "普通放大模型":
                if deferred_decode:
                    raise RuntimeError("统一解码模式下不能使用普通视频放大模型：请改用 H3 Latent 学习型放大或关闭统一解码")
                preview = _upscale_with_model(upscale_model, decoded)
                preview = _upscale_interpolate(preview, width, height)
            elif mode == "自动（RTX→普通模型→插值）":
                if deferred_decode:
                    raise RuntimeError("统一解码模式下不能使用视频放大链：请改用 H3 Latent 学习型放大或关闭统一解码")
                try:
                    preview = _upscale_rtx(decoded, width, height, "高")
                except Exception:
                    preview = _upscale_with_model(upscale_model, decoded) if upscale_model else decoded
                    preview = _upscale_interpolate(preview, width, height)
            else:
                if deferred_decode:
                    raise RuntimeError("统一解码模式下不能使用像素插值放大：请改用 H3 Latent 学习型放大或关闭统一解码")
                preview = _upscale_interpolate(decoded, width, height)
            # Some VAE/upscaler combinations round the temporal dimension while
            # processing a video.  Restore H3's exact 17k+5 frame count before
            # the second VAE encode; spatial scaling must never change time.
            if expected_frames is not None:
                preview = _match_video_frame_count(preview, expected_frames)
            if mode != "H3 Latent 学习型放大":
                encoded_video = _encode_h3_video(video_vae, preview)
                encoded_video = _match_latent_time(encoded_video, first_video.shape[2])
        # The two branches can be returned on different devices/dtypes by
        # custom VAEs.  Normalize audio before rebuilding H3's AV container.
        # Keep the H3 audio stream in its native latent layout.  Casting an
        # audio latent to a custom video VAE dtype can quantize it to bf16/fp16
        # and is a common source of audible broadband distortion.
        first_audio = _normalize_h3_audio_latent(first_audio, "第一阶段音频 latent")
        first_audio = first_audio.to(device=encoded_video.device)
        refined = dict(first)
        refined["samples"] = _h3_av_container(encoded_video, first_audio)
        # Context sampled-noise masking belongs only to the first pass. The
        # second pass starts from the completed first-pass latent and must not
        # reapply its prefix mask at a different spatial resolution.
        refined.pop("noise_mask", None)
        refined.pop("h3_auto_director_sampled_context", None)
        # Retain text and user references, but remove the project continuation
        # injected for stage one unless the user explicitly asks to apply it
        # again during refinement. This also applies to a separately connected
        # stage-two conditioning if it came from our Motion Context adapter.
        # When provided, this is the separately saved final (second-pass)
        # context from the previous segment. It is deliberately independent
        # of the first-pass context used to create ``conditioning``.
        stage2_source = stage2_context_latent
        if isinstance(stage2_source, dict) and stage2_source.get("h3_stage2_context_latent") is not None:
            stage2_source = stage2_source["h3_stage2_context_latent"]
        # Optional stage-two conditioning sockets are often left connected to
        # an empty cache/placeholder node in older workflows.  An empty list
        # must not replace the valid first-stage conditioning, otherwise the
        # second sampler fails after stage one has already completed.
        stage2_input_conditioning = (
            stage2_conditioning if _has_positive_conditioning(stage2_conditioning)
            else conditioning
        )
        stage2_input_conditioning = _attach_union_control_conditioning(stage2_input_conditioning, control_config)
        final_conditioning = _prepare_stage2_conditioning(
            stage2_input_conditioning, refined, stage2_use_context, stage2_source
        )
        if not _has_positive_conditioning(final_conditioning):
            raise ValueError(
                "二采正向条件为空：请确认多模态参考节点已输出 positive，"
                "且不要把空的 stage2_conditioning 节点连接到双采样。"
            )
        # A single external schedule can drive both passes.  When a dedicated
        # second-pass SIGMAS socket is not connected, reuse the first-pass
        # schedule instead of silently falling back to a different scheduler.
        effective_stage2_sigmas = stage2_sigmas if stage2_sigmas is not None else stage1_sigmas
        if stage2_sigmas is None and stage1_sigmas is not None:
            LOG.info("H3 Auto Director: 二采 Sigmas 未连接，复用一采 Sigmas")
        final = _dual_sample(second_model, final_conditioning, refined, sampler_name, scheduler,
                             stage2_steps, stage2_denoise, int(seed) + 1, enable_preview,
                             effective_stage2_sigmas, stage2_extend_sigmas, stage2_extend_steps,
                             stage2_start_at_sigma, stage2_end_at_sigma, stage2_spacing)
        if bool(use_stage1_audio_only):
            final_parts = _av_latent_parts(final)
            if final_parts is None:
                raise ValueError("二采输出不是 H3 联合 AV latent，无法替换为一采音频")
            final_audio = _normalize_h3_audio_latent(first_audio, "一采音频 latent")
            final_audio = final_audio.to(device=final_parts[0].device)
            final = dict(final)
            final["samples"] = _h3_av_container(final_parts[0], final_audio)
            if stage1_audio_override is None:
                raise RuntimeError("一采音频已启用，但未生成音频覆盖数据")
            final["_h3_stage1_audio"] = stage1_audio_override
            LOG.info("H3 Auto Director: 已启用最终仅使用一采音频，二采仅输出画面")
        final_output = dict(final)
        final_output["_h3_stage1_context"] = first
        return (final_output, first, preview, decoded)


class H3AutoDirectorDualSamplingModel:
    """Graph-friendly dual sampler that accepts the existing H3 model chain."""

    @classmethod
    def INPUT_TYPES(cls):
        base = {"required": {
            "stage1_model": ("MODEL", {"tooltip": "一采模型。可直接连接模型加载器或外部 LoRA/显存优化链。"}),
            "conditioning": ("CONDITIONING",), "latent": ("LATENT",),
            "video_vae": ("VAE",), "audio_vae": ("VAE",),
            "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "res_multistep"}),
            "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
            "stage1_steps": ("INT", {"default": 6, "min": 1, "max": 100}),
            "stage1_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_stage2": ("BOOLEAN", {"default": True, "label_on": "启用二采", "label_off": "关闭二采",
                               "tooltip": "关闭后仅执行第一阶段采样。"}),
            "stage2_use_context": ("BOOLEAN", {"default": False, "label_on": "二采使用上下文（实验性）", "label_off": "二采不使用上下文",
                                    "tooltip": "实验性功能，当前效果不可用，默认应关闭。开启后会尝试将上一段最终二采视频/音频上下文适配到二采尺寸后传入。"}),
            "stage2_steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "stage2_denoise": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            "upscale_mode": (DUAL_UPSCALE_CHOICES, {"default": "普通插值"}),
            "target_width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32, "forceInput": True}),
            "target_height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32, "forceInput": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "use_stage1_audio_only": ("BOOLEAN", {"default": False,
                                      "label_on": "最终仅使用一采音频",
                                      "label_off": "使用二采音频",
                                      "tooltip": "二采继续细化画面，但最终 AV latent 使用第一阶段生成的音频。"}),
            "enable_preview": ("BOOLEAN", {"default": False, "label_on": "开启新版采样预览", "label_off": "关闭新版采样预览"}),
            "latent_upscale_model": ((_h3_latent_upscaler.available_models() if _h3_latent_upscaler else ["(未加载 H3 latent 放大器)"]), {"default": (_h3_latent_upscaler.available_models()[0] if _h3_latent_upscaler and _h3_latent_upscaler.available_models() else "")}),
            "latent_upscale_device": (["cuda", "cpu"], {"default": "cuda"}),
            "latent_upscale_precision": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
            "stage1_extend_sigmas": ("BOOLEAN", {"default": False, "label_on": "开启一采 Sigmas 扩展", "label_off": "关闭一采 Sigmas 扩展", "tooltip": "开启后按 ComfyUI ExtendIntermediateSigmas 原版算法扩展一采调度。"}),
            "stage1_extend_steps": ("INT", {"default": 2, "min": 1, "max": 100}),
            "stage1_start_at_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20000.0, "step": 0.01, "round": False}),
            "stage1_end_at_sigma": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 20000.0, "step": 0.01, "round": False}),
            "stage1_spacing": (["linear", "cosine", "sine"], {"default": "linear"}),
            "stage2_extend_sigmas": ("BOOLEAN", {"default": False, "label_on": "开启二采 Sigmas 扩展", "label_off": "关闭二采 Sigmas 扩展", "tooltip": "开启后按 ComfyUI ExtendIntermediateSigmas 原版算法扩展二采调度。"}),
            "stage2_extend_steps": ("INT", {"default": 2, "min": 1, "max": 100}),
            "stage2_start_at_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20000.0, "step": 0.01, "round": False}),
            "stage2_end_at_sigma": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 20000.0, "step": 0.01, "round": False}),
            "stage2_spacing": (["linear", "cosine", "sine"], {"default": "linear"}),
        }, "optional": {
            "upscale_model": ("UPSCALE_MODEL",),
            "stage2_model": ("MODEL", {"tooltip": "可选。连接外部 LoRA/显存优化后的第二阶段模型；未连接时自动复用一采模型。"}),
            "audio_sampling": ("H3_AUDIO_SAMPLING", {"tooltip": "可选。连接‘音频采样切换’的采样调度信息；它只设置 H3 音频采样方法与偏移，不会覆盖两阶段的步数、降噪或调度器。"}),
            "stage1_sigmas": ("SIGMAS", {"tooltip": "可选。一采使用的完整 Sigmas 调度；连接后优先于一采步数、降噪和调度器。"}),
            "stage2_sigmas": ("SIGMAS", {"tooltip": "可选。二采使用的完整 Sigmas 调度；连接后优先于二采步数、降噪和调度器。"}),
            "stage2_conditioning": ("CONDITIONING", {"tooltip": "可选。二采专用正向条件；未连接时复用一采条件。"}),
            "stage2_context_latent": ("LATENT", {"tooltip": "可选。二采上下文潜变量；仅在开启二采上下文时使用。"}),
            "control_config": ("H3_CONTROL_CONFIG", {"tooltip": "可选。连接统一姿态/深度控制节点；控制视频路径与独立权重会传入采样条件。"}),
        }}
        return base

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("最终 AV latent", "第一阶段 AV latent", "放大预览", "第一阶段画面")
    FUNCTION = "sample"
    CATEGORY = "H3 自动导演/采样"

    def sample(self, stage1_model, conditioning, latent, video_vae, audio_vae, sampler_name, scheduler,
               stage1_steps, stage1_denoise, stage2_steps, stage2_denoise, upscale_mode,
               target_width, target_height, enable_stage2=True, stage2_use_context=False,
               seed=0, upscale_model=None, stage2_conditioning=None, stage2_context_latent=None,
               use_stage1_audio_only=False, enable_preview=False, latent_upscale_model=None,
               latent_upscale_device="cuda", latent_upscale_precision="fp32",
               stage2_model=None, audio_sampling=None, stage1_sigmas=None, stage2_sigmas=None,
               stage1_extend_sigmas=False, stage1_extend_steps=2, stage1_start_at_sigma=-1.0,
               stage1_end_at_sigma=12.0, stage1_spacing="linear", stage2_extend_sigmas=False,
               stage2_extend_steps=2, stage2_start_at_sigma=-1.0, stage2_end_at_sigma=12.0,
               stage2_spacing="linear", control_config=None, **_legacy_unused):
        return H3AutoDirectorDualSampling().sample(
        stage1_model, conditioning, latent, video_vae, audio_vae, sampler_name, scheduler,
            stage1_steps, stage1_denoise, stage2_steps, stage2_denoise, upscale_mode,
            target_width, target_height, enable_stage2, stage2_use_context, upscale_model, seed,
            stage2_conditioning, stage2_context_latent, use_stage1_audio_only,
            enable_preview, latent_upscale_model, latent_upscale_device,
            latent_upscale_precision, stage2_model, audio_sampling, stage1_sigmas, stage2_sigmas,
            stage1_extend_sigmas, stage1_extend_steps, stage1_start_at_sigma, stage1_end_at_sigma,
            stage1_spacing, stage2_extend_sigmas, stage2_extend_steps, stage2_start_at_sigma,
            stage2_end_at_sigma, stage2_spacing, control_config=control_config)


def _validate_reference_limits(refs, label="参考素材"):
    """Validate H3 per-segment reference limits before any files are loaded."""
    if not isinstance(refs, list):
        raise ValueError(f"{label}必须是 JSON 列表")
    valid = [ref for ref in refs if isinstance(ref, dict) and (
        ref.get("path") or ref.get("name") or str(ref.get("type", "")).lower() == "previous_segment_video"
    )]
    if len(valid) > MAX_REFERENCE_TOTAL:
        raise ValueError(f"{label}总数最多 {MAX_REFERENCE_TOTAL} 个")
    counts = {"image": 0, "video": 0, "audio": 0}
    for ref in valid:
        kind = str(ref.get("type", "image")).lower()
        if kind in {"previous_segment_video", "transfer_video_segment"}:
            kind = "video"
        if kind not in counts:
            raise ValueError("未知参考素材类型: %s" % kind)
        counts[kind] += 1
    if counts["image"] > MAX_REFERENCE_IMAGES:
        raise ValueError(f"{label}图片最多 {MAX_REFERENCE_IMAGES} 个")
    if counts["video"] > MAX_REFERENCE_VIDEOS:
        raise ValueError(f"{label}视频最多 {MAX_REFERENCE_VIDEOS} 个")
    if counts["audio"] > MAX_REFERENCE_AUDIOS:
        raise ValueError(f"{label}独立音频最多 {MAX_REFERENCE_AUDIOS} 个")


class H3AutoDirectorPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_id": ("STRING", {"default": "h3_project"}),
            "segments_json": ("STRING", {"default": '[{"prompt":"","duration":5,"audio_restart":false}]', "multiline": True}),
            "duration": ("FLOAT", {"default": 5.0, "min": 4.0, "max": 15.0, "step": 0.1}),
            "global_reference_set": ("BOOLEAN", {"default": True}),
            "auto_run": ("BOOLEAN", {"default": True}),
            "continuation_mode": ("BOOLEAN", {"default": True, "tooltip": "默认允许后续片段使用视频上下文；每段可单独关闭"}),
            "cache_prompt_embeddings": ("BOOLEAN", {"default": False, "tooltip": "首次执行时一次性编码并缓存全部片段的多模态提示词向量"}),
            "decode_after_all_segments": ("BOOLEAN", {"default": False,
                "tooltip": "开启后先缓存每段最终 AV latent，全部采样完成后统一解码、裁剪并拼接；视频上下文会自动改为缓存潜空间直取。"}),
            "output_root": ("STRING", {"default": "h3_projects", "tooltip": "项目文件夹名称；新路径为 output/h3_project/<此名称>"}),
            "cache_prompt_embeddings_to_disk": ("BOOLEAN", {"default": False, "tooltip": "将提示词向量保存到项目 cache/prompt_embeddings；清单 JSON 会按提示词、素材和编码器配置判断是否重新编码"}),
        }, "optional": {
            "global_assets_json": ("STRING", {"default": "[]", "multiline": True}),
            "auto_context_crop_frames": ("INT", {"default": 0, "min": 0, "max": 4096,
                "tooltip": "自动裁剪上下文时使用的帧数；0 表示按上下文长度自动计算；大于 0 时自动启用裁剪。"}),
        }, "hidden": {"project_dir": "STRING"}}

    RETURN_TYPES = ("H3_AUTO_PLAN",)
    RETURN_NAMES = ("项目计划",)
    FUNCTION = "create"
    CATEGORY = "H3 自动导演"

    def create(self, project_id, segments_json, duration, global_reference_set, auto_run, continuation_mode=True, cache_prompt_embeddings=False, decode_after_all_segments=False, output_root="h3_projects", cache_prompt_embeddings_to_disk=False, global_assets_json="[]", auto_context_crop_frames=0, project_dir="", **_legacy_inputs):
        try:
            segments = json.loads(segments_json)
            assets = json.loads(global_assets_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError("segments_json/global_assets_json must be valid JSON: %s" % exc) from exc
        if not isinstance(segments, list) or not segments:
            raise ValueError("At least one H3 segment is required")
        if not isinstance(assets, list):
            raise ValueError("global_assets_json must be a JSON list")
        try:
            auto_context_crop_frames = max(0, min(4096, int(auto_context_crop_frames or 0)))
        except (TypeError, ValueError) as exc:
            raise ValueError("自动裁剪上下文帧数必须是 0 到 4096 的整数") from exc
        normalized = []
        for item in segments:
            if not isinstance(item, dict):
                raise ValueError("Each segment must be an object")
            row = dict(item)
            row["prompt"] = str(row.get("prompt", "")).strip()
            row["duration"] = float(row.get("duration", duration))
            if not 4.0 <= row["duration"] <= 15.0:
                raise ValueError("H3 segment duration must be between 4 and 15 seconds")
            row["audio_restart"] = bool(row.get("audio_restart", False))
            row["continue_audio"] = bool(row.get("continue_audio", True))
            row["continue_video"] = bool(row.get("continue_video", bool(continuation_mode) and len(normalized) > 0))
            if "use_previous_video_reference" in row:
                row["use_previous_video_reference"] = bool(row["use_previous_video_reference"])
            # Reference images condition the full clip; H3 has no per-image
            # duration input, so discard this legacy UI field on every save.
            row.pop("image_duration", None)
            raw_references = row.get("references", [])
            if not isinstance(raw_references, list):
                raise ValueError("每段参考素材必须是 JSON 列表")
            references = []
            for reference in raw_references:
                if not isinstance(reference, dict):
                    continue
                reference = dict(reference)
                try:
                    reference["insert_seconds"] = max(0.0, float(reference.get("insert_seconds", 0) or 0))
                    reference["insert_frames"] = max(0, int(reference.get("insert_frames", 0) or 0))
                except (TypeError, ValueError) as exc:
                    raise ValueError("素材插入时间必须是非负秒数和帧数") from exc
                references.append(reference)
            row["references"] = references
            _validate_reference_limits(row["references"], f"第 {len(normalized) + 1} 段参考素材")
            normalized.append(row)
        # The UI keeps this field in sync for compatibility, but the global
        # policy is deliberately defined by the first segment itself. This
        # prevents stale hidden JSON in an older workflow from overriding it.
        if global_reference_set:
            assets = list(normalized[0].get("references", []))
        _validate_reference_limits(assets, "统一参考素材")
        for index, row in enumerate(normalized, start=1):
            if index <= 1 or not bool(row.get("use_previous_video_reference", False)):
                continue
            effective_refs = assets if global_reference_set else row["references"]
            video_count = sum(
                1 for ref in effective_refs
                if isinstance(ref, dict) and str(ref.get("type", "image")).lower() == "video"
            )
            if video_count > 1:
                raise ValueError(f"第 {index} 段启用上片段视频参考时，最多只能上传 1 个视频参考素材")
        requested_dir = str(project_dir or "").strip().strip('"')
        if requested_dir:
            project_dir = Path(requested_dir).expanduser().resolve()
            if _output_root() not in project_dir.parents:
                raise ValueError("项目目录必须位于 ComfyUI output 内")
        else:
            project_dir = _safe_project_dir(project_id, output_root)
        # Keep an old project as a read-only continuation fallback while new
        # clips are written into the new layout.
        legacy_project_dir = _find_legacy_project(project_id, output_root, project_dir)
        plan = {"version": 2, "project_id": project_id, "output_root": output_root,
                "duration": float(duration), "global_reference_set": bool(global_reference_set),
                "auto_run": bool(auto_run), "continuation_mode": bool(continuation_mode),
                "cache_prompt_embeddings": bool(cache_prompt_embeddings),
                "decode_after_all_segments": bool(decode_after_all_segments),
                "cache_prompt_embeddings_to_disk": bool(cache_prompt_embeddings_to_disk),
                "auto_context_crop_frames": int(auto_context_crop_frames),
                "global_assets": assets, "segments": normalized,
                "project_dir": str(project_dir)}
        if legacy_project_dir is not None:
            plan["legacy_project_dir"] = str(legacy_project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "json").mkdir(exist_ok=True)
        (project_dir / "cache").mkdir(exist_ok=True)
        (project_dir / CONTEXT_DIR_NAME).mkdir(exist_ok=True)
        (project_dir / "clips").mkdir(exist_ok=True)
        (project_dir / "final").mkdir(exist_ok=True)
        _atomic_json(project_dir / "json" / "project.json", {k: v for k, v in plan.items() if k != "project_dir"})
        state = _load_json(_state_path(plan), {"version": 2, "segments": {}})
        state.setdefault("segments", {})
        _atomic_json(project_dir / "json" / "state.json", state)
        return (plan,)


class H3AutoDirectorTTSPlan:
    """Audio-focused H3 plan with per-segment multimodal references."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_id": ("STRING", {"default": "h3_tts_project"}),
            "segments_json": ("STRING", {"default": '[{"prompt":"","duration":5,"audio_filename":""}]', "multiline": True}),
            "auto_run": ("BOOLEAN", {"default": True}),
            "cache_prompt_embeddings": ("BOOLEAN", {"default": True}),
            "enable_audio_continuation": ("BOOLEAN", {"default": True}),
            "concat_final_audio": ("BOOLEAN", {"default": True, "tooltip": "完成全部片段后额外拼接一个长 WAV；关闭则只保留分段音频"}),
            "output_root": ("STRING", {"default": "h3_tts_project"}),
            "global_reference_set": ("BOOLEAN", {"default": False, "tooltip": "开启后所有片段使用第 1 段的图片、视频和音频参考素材"}),
            "cache_prompt_embeddings_to_disk": ("BOOLEAN", {"default": False, "tooltip": "将提示词向量保存到项目 cache/prompt_embeddings；清单 JSON 会按提示词、素材和编码器配置判断是否重新编码"}),
        }, "hidden": {
            "project_dir": "STRING",
            # Legacy fields are accepted by create() when loading old plans,
            # but are intentionally no longer exposed as node inputs.
            "reference_video_json": "STRING", "reference_assets_json": "STRING",
            "pass_reference_video_audio": "BOOLEAN", "audio_restart_segments": "STRING",
        }}

    RETURN_TYPES = ("H3_AUTO_PLAN",)
    RETURN_NAMES = ("TTS 项目计划",)
    FUNCTION = "create"
    CATEGORY = "H3 自动导演/TTS"

    def create(self, project_id, segments_json, auto_run=True, cache_prompt_embeddings=True,
               enable_audio_continuation=True, concat_final_audio=True,
               output_root="h3_tts_project", global_reference_set=False,
               cache_prompt_embeddings_to_disk=False,
               project_dir="", reference_video_json="{}", reference_assets_json="[]",
               pass_reference_video_audio=False, audio_restart_segments="", **_legacy_inputs):
        try:
            rows = json.loads(segments_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError("TTS 片段 JSON 无效") from exc
        try:
            reference_video = json.loads(reference_video_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("TTS 参考视频 JSON 无效") from exc
        if not isinstance(reference_video, dict):
            reference_video = {}
        if reference_video.get("path") or reference_video.get("name"):
            _reference_name(reference_video)
        try:
            reference_assets = json.loads(reference_assets_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError("TTS 参考素材 JSON 无效") from exc
        if not isinstance(reference_assets, list):
            raise ValueError("TTS 参考素材必须是列表")
        for asset in reference_assets:
            if not isinstance(asset, dict) or str(asset.get("type", "")).lower() not in {"image", "video", "audio"}:
                raise ValueError("TTS 参考素材仅支持图片、视频和音频")
        # Older TTS plans stored one global asset list and one global video.
        # Keep accepting that shape, but new plans store references per segment.
        legacy_references = list(reference_assets)
        if reference_video.get("path") or reference_video.get("name"):
            legacy_references.append({
                "type": "video", "path": reference_video.get("path") or reference_video.get("name"),
                "name": reference_video.get("name") or Path(str(reference_video.get("path"))).name,
                "video_audio_enabled": bool(pass_reference_video_audio),
            })
        _validate_reference_limits(legacy_references, "TTS 旧版参考素材")
        if not isinstance(rows, list) or not rows:
            raise ValueError("至少需要一个 TTS 片段")
        restart_at = _parse_segment_numbers(audio_restart_segments, "重新生成音频片段")
        if any(index > len(rows) for index in restart_at):
            raise ValueError("重新生成音频片段编号超出片段总数")
        normalized = []
        filenames = set()
        for index, item in enumerate(rows, start=1):
            if not isinstance(item, dict):
                raise ValueError("每个 TTS 片段必须是对象")
            duration = float(item.get("duration", 5.0))
            if not 4.0 <= duration <= 15.0:
                raise ValueError("TTS 片段时长必须在 4 到 15 秒之间")
            audio_filename = _audio_filename(item.get("audio_filename", ""), index)
            if audio_filename.casefold() in filenames:
                raise ValueError("TTS 每段音频文件名不能重复：%s" % audio_filename)
            filenames.add(audio_filename.casefold())
            has_row_references = "references" in item
            row_references = item.get("references")
            if not has_row_references:
                # A legacy row has no references field. Migrate the old global
                # collection into every row. An explicit [] in a new plan is
                # meaningful and must remain empty when unified references are
                # disabled, even though the compatibility field is populated.
                row_references = copy.deepcopy(legacy_references)
            elif not isinstance(row_references, list):
                raise ValueError("TTS 每段参考素材必须是列表")
            for ref in row_references:
                if not isinstance(ref, dict) or str(ref.get("type", "")).lower() not in {"image", "video", "audio"}:
                    raise ValueError("TTS 每段参考素材仅支持图片、视频和音频")
            _validate_reference_limits(row_references, f"TTS 第 {index} 段参考素材")
            normalized.append({
                "prompt": str(item.get("prompt", "")).strip(),
                "duration": duration,
                "audio_filename": audio_filename,
                "audio_restart": bool(item.get("audio_restart", False)) or index in restart_at,
                "continue_audio": bool(item.get("continue_audio", True)),
                "continue_video": False,
                "references": copy.deepcopy(row_references),
            })
        unified_references = copy.deepcopy(normalized[0]["references"]) if global_reference_set else []
        if global_reference_set:
            _validate_reference_limits(unified_references, "TTS 统一参考素材")
            for row in normalized:
                row["references"] = copy.deepcopy(unified_references)
        requested_dir = str(project_dir or "").strip().strip('"')
        directory = Path(requested_dir).expanduser().resolve() if requested_dir else _safe_project_dir(project_id, output_root)
        if requested_dir and _output_root() not in directory.parents:
            raise ValueError("项目目录必须位于 ComfyUI output 内")
        # TTS never writes video clips or visual context.  Keep its generated
        # pieces isolated from the final mix while retaining old layouts on
        # read through _audio_path().
        for name in ("json", "cache", "audio/segments", "final"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        plan = {
            "version": 3, "mode": "tts", "project_id": str(project_id),
            "output_root": str(output_root), "duration": 5.0,
            "global_reference_set": bool(global_reference_set), "global_assets": unified_references,
            "auto_run": bool(auto_run), "continuation_mode": bool(enable_audio_continuation),
            "video_continuation": False, "cache_prompt_embeddings": bool(cache_prompt_embeddings),
            "cache_prompt_embeddings_to_disk": bool(cache_prompt_embeddings_to_disk),
            "segments": normalized, "project_dir": str(directory),
            "concat_final_audio": bool(concat_final_audio),
            "reference_video": reference_video,
            "reference_assets": reference_assets,
        }
        _atomic_json(directory / "json" / "project.json", {k: v for k, v in plan.items() if k != "project_dir"})
        state = _load_json(_state_path(plan), {"version": 3, "segments": {}})
        state.setdefault("segments", {})
        _atomic_json(_state_path(plan), state)
        return (plan,)


class H3AutoDirectorVideoTransferPlan:
    """Build a fixed-prompt H3 transfer plan from one reference video timeline.

    The reference video's 24 fps timeline is partitioned once, then each
    generated segment receives the matching source-video window as Video 1.
    The final short window is tail-frame padded only for reference encoding;
    its requested generation duration remains the actual remainder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_id": ("STRING", {"default": "h3_video_transfer"}),
            "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
            "reference_video_json": ("STRING", {"default": "{}", "multiline": True}),
            "reference_assets_json": ("STRING", {"default": "[]", "multiline": True}),
            "segment_seconds": ("FLOAT", {"default": 5.0, "min": 4.0, "max": 15.0, "step": 0.1}),
            "pass_reference_video_audio": ("BOOLEAN", {"default": False}),
            "enable_audio_continuation": ("BOOLEAN", {"default": True}),
            "audio_restart_segments": ("STRING", {"default": "", "tooltip": "片段编号，从 1 开始；支持中英文逗号，例如 3，6,9"}),
            "previous_video_reference_segments": ("STRING", {"default": "", "tooltip": "片段编号，从 2 开始；该段额外使用上段生成视频画面"}),
            "cache_prompt_embeddings": ("BOOLEAN", {"default": True}),
            "skip_h3_audio_decode": ("BOOLEAN", {"default": False, "tooltip": "仅跳过 H3 音频 VAE 解码；H3 的联合 AV 采样仍会执行"}),
            "final_audio_source": (["H3 生成音频", "参考视频音频"], {"default": "H3 生成音频"}),
            "auto_run": ("BOOLEAN", {"default": True}),
            "output_root": ("STRING", {"default": "h3_video_transfer"}),
            "cache_prompt_embeddings_to_disk": ("BOOLEAN", {"default": False, "tooltip": "将提示词向量保存到项目 cache/prompt_embeddings；清单 JSON 会按提示词、素材和编码器配置判断是否重新编码"}),
        }, "hidden": {"project_dir": "STRING"}}

    RETURN_TYPES = ("H3_AUTO_PLAN",)
    RETURN_NAMES = ("动作迁移项目计划",)
    FUNCTION = "create"
    CATEGORY = "H3 自动导演/视频迁移"

    def create(self, project_id, prompt, reference_video_json, reference_assets_json,
               segment_seconds, pass_reference_video_audio=False,
               enable_audio_continuation=True, audio_restart_segments="",
               previous_video_reference_segments="", cache_prompt_embeddings=True,
               skip_h3_audio_decode=False, final_audio_source="H3 生成音频",
               auto_run=True, output_root="h3_video_transfer", cache_prompt_embeddings_to_disk=False,
               project_dir=""):
        try:
            video = json.loads(reference_video_json or "{}")
            assets = json.loads(reference_assets_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError("视频迁移素材 JSON 无效: %s" % exc) from exc
        if not isinstance(video, dict) or not (video.get("path") or video.get("name")):
            raise ValueError("请在“编辑视频迁移素材”中上传一个参考视频")
        _reference_name(video)
        if not isinstance(assets, list):
            raise ValueError("图片与独立音频参考素材必须是 JSON 列表")
        for asset in assets:
            kind = str(asset.get("type", "")).lower() if isinstance(asset, dict) else ""
            if kind not in {"image", "audio"}:
                raise ValueError("视频迁移节点的附加参考素材仅支持图片和独立音频；参考视频请在专用上传项中选择")
        _validate_reference_limits([{"type": "video"}] + assets, "视频迁移参考素材")
        # These controls serve different stages.  ``pass_reference_video_audio``
        # adds the source audio to H3's per-segment multimodal conditioning,
        # while ``final_audio_source`` muxes the original full source track
        # only after the generated video has been assembled.  Selecting the
        # latter must not force source audio into the H3 conditioning path.
        seconds = float(segment_seconds)
        if not 4.0 <= seconds <= 15.0:
            raise ValueError("单段秒数必须在 4 到 15 秒之间")
        # Browser probing supplies the exact decoded 24 fps count. Fall back
        # to duration for imported projects created without the editor.
        source_frames = int(video.get("frame_count_24") or 0)
        if source_frames <= 0:
            source_frames = int(round(float(video.get("duration", 0)) * FPS))
        if source_frames < 5:
            raise ValueError("参考视频时长不足，至少需要 5 帧")
        per_segment_frames = max(1, int(round(seconds * FPS)))
        restart_at = _parse_segment_numbers(audio_restart_segments, "重新生成音频片段")
        previous_ref_at = _parse_segment_numbers(previous_video_reference_segments, "上段视频参考片段")
        count = int(math.ceil(source_frames / per_segment_frames))
        invalid = sorted(number for number in restart_at | previous_ref_at if number > count)
        if invalid:
            raise ValueError("片段编号超出参考视频范围：%s（共 %d 段）" % (", ".join(map(str, invalid)), count))
        if 1 in previous_ref_at:
            raise ValueError("第 1 段没有上段生成视频，不能启用上段视频参考")

        normalized = []
        for index in range(1, count + 1):
            start_frame = (index - 1) * per_segment_frames
            available = min(per_segment_frames, source_frames - start_frame)
            # H3 reference-video frames use the 17k+5 grid. Pad by repeating
            # the actual tail frame instead of dropping the final motion.
            reference_frames = _align_frames(available)
            transfer_ref = {
                "type": "transfer_video_segment", "path": video.get("path") or video.get("name"),
                "name": video.get("name") or Path(str(video.get("path"))).name,
                "start_frame": start_frame, "source_frames": available,
                "reference_frames": reference_frames,
                "video_audio_enabled": bool(pass_reference_video_audio),
            }
            normalized.append({
                "prompt": str(prompt or "").strip(), "duration": available / FPS,
                "audio_restart": index in restart_at,
                # Transfer uses the matching source window for motion. In
                # addition, keep generated-scene continuity by default from
                # segment 2 onward. A segment listed in
                # ``previous_video_reference_segments`` switches to the
                # previous generated clip as a multimodal video reference and
                # therefore disables pixel/video context for that segment.
                "continue_video": index > 1 and index not in previous_ref_at,
                "use_previous_video_reference": index in previous_ref_at,
                "references": [transfer_ref] + [dict(item) for item in assets],
            })
        requested_dir = str(project_dir or "").strip().strip('"')
        directory = Path(requested_dir).expanduser().resolve() if requested_dir else _safe_project_dir(project_id, output_root)
        if requested_dir and _output_root() not in directory.parents:
            raise ValueError("项目目录必须位于 ComfyUI output 内")
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("json", "cache", CONTEXT_DIR_NAME, "clips", "final"):
            (directory / name).mkdir(exist_ok=True)
        plan = {
            "version": 3, "mode": "video_transfer", "project_id": project_id,
            "output_root": output_root, "duration": seconds,
            "global_reference_set": False, "global_assets": [],
            "auto_run": bool(auto_run),
            # Audio and video context policies are intentionally separate.
            "continuation_mode": bool(enable_audio_continuation),
            # Video context continuation is enabled by default for transfer.
            # The per-segment previous-video-reference flag is the explicit
            # opt-out for selected segments.
            "video_continuation": True,
            "cache_prompt_embeddings": bool(cache_prompt_embeddings),
            "cache_prompt_embeddings_to_disk": bool(cache_prompt_embeddings_to_disk),
            "skip_h3_audio_decode": bool(skip_h3_audio_decode),
            "final_audio_source": str(final_audio_source),
            "reference_video": dict(video), "segments": normalized,
            "project_dir": str(directory),
        }
        _atomic_json(directory / "json" / "project.json", {k: v for k, v in plan.items() if k != "project_dir"})
        state = _load_json(_state_path(plan), {"version": 3, "segments": {}})
        state.setdefault("segments", {})
        _atomic_json(directory / "json" / "state.json", state)
        return (plan,)


CONTROL_MODES = ("关闭", "姿态", "深度", "姿态+深度")


def _control_image_result(result):
    """Normalize ControlNet-Aux node return values to an IMAGE tensor."""
    if isinstance(result, dict):
        result = result.get("result", ())
    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    if not torch.is_tensor(result) or result.ndim != 4:
        raise ValueError("控制预处理器没有输出 [帧,高,宽,RGB] 图像序列")
    if result.shape[-1] == 1:
        result = result.repeat(1, 1, 1, 3)
    if result.shape[-1] < 3:
        raise ValueError("控制预处理器输出通道数不足")
    return result[..., :3].float().clamp(0, 1).contiguous()


def _align_control_frames(images):
    """Align a control batch to the H3 17*n+5 temporal grid by tail repeat."""
    count = int(images.shape[0])
    if count < 5:
        raise ValueError("H3 控制视频至少需要 5 帧")
    aligned = _align_frames(count)
    if count < aligned:
        images = torch.cat((images, images[-1:].repeat((aligned - count, 1, 1, 1))), dim=0)
    return images[:aligned], aligned


def _control_cache_dir(plan, segment_index):
    """Return the per-project directory for persisted control frames."""
    if not isinstance(plan, dict) or not plan.get("project_dir"):
        return None
    directory = Path(plan["project_dir"]) / "control" / ("segment_%04d" % int(segment_index))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _control_cache_paths(plan, segment_index, kind):
    directory = _control_cache_dir(plan, segment_index)
    if directory is None:
        return None, None
    return directory / ("%s.pt" % kind), directory / ("%s.mp4" % kind)


def _load_control_tensor(path):
    if not path or not Path(path).is_file():
        return None
    try:
        value = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        LOG.warning("H3 Auto Director: 控制帧缓存读取失败，将重新预处理：%s", exc)
        return None
    if not torch.is_tensor(value) or value.ndim != 4:
        return None
    return value.float().clamp(0, 1).contiguous()


def _save_control_tensor(path, images):
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(images.detach().float().cpu().contiguous(), str(temporary))
    temporary.replace(path)


def _control_mode_has_pose(mode):
    return str(mode or "").strip() in {"姿态", "姿态+深度", "pose", "pose+depth"}


def _control_mode_has_depth(mode):
    return str(mode or "").strip() in {"深度", "姿态+深度", "depth", "pose+depth"}


def _control_frames_for_segment(plan, segment_index, video_frames):
    """Slice a full reference-video IMAGE batch to one transfer segment.

    When a caller already supplies a segment-sized batch, it is returned as-is.
    This lets the node work with either a normal video loader or the plan-only
    automatic loader without duplicating a source video in the graph.
    """
    frames = video_frames
    if not isinstance(plan, dict) or str(plan.get("mode", "")) != "video_transfer":
        return frames
    try:
        segment = _segment(plan, max(1, int(segment_index)))
        transfer = next(item for item in segment.get("references", [])
                        if isinstance(item, dict) and item.get("type") == "transfer_video_segment")
    except Exception:
        return frames
    start = max(0, int(transfer.get("start_frame", 0)))
    source = max(1, int(transfer.get("source_frames", transfer.get("reference_frames", 1))))
    aligned = max(source, int(transfer.get("reference_frames", source)))
    count = int(frames.shape[0])
    if start > 0 and count >= start + source:
        return frames[start:start + source]
    if start == 0 and count > aligned:
        return frames[:source]
    return frames


def _load_transfer_source_frames(plan, segment_index):
    segment = _segment(plan, max(1, int(segment_index)))
    transfer = next((item for item in segment.get("references", [])
                     if isinstance(item, dict) and item.get("type") == "transfer_video_segment"), None)
    if transfer is None:
        raise ValueError("动作迁移计划缺少参考视频片段")
    return _load_transfer_video_segment(transfer)


def _control_preprocessor_mappings():
    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    for package_name in ("comfyui_controlnet_aux", "custom_nodes.comfyui_controlnet_aux"):
        try:
            aux_module = importlib.import_module(package_name)
            aux_mappings = getattr(aux_module, "AUX_NODE_MAPPINGS", None)
            if aux_mappings:
                mappings = {**aux_mappings, **mappings}
                break
        except Exception:
            continue
    return mappings


def _run_pose_preprocessor(video_frames, resolution, mappings):
    # DWPose is the most reliable common denominator for real and anime
    # footage.  An AnimePose node is used only when DWPose is unavailable.
    cls = next((mappings.get(name) for name in ("DWPreprocessor", "OpenposePreprocessor",
                                                "AnimePosePreprocessor", "AnimePose")
                if mappings.get(name) is not None), None)
    if cls is None:
        raise RuntimeError("未找到姿态预处理器，请安装 comfyui_controlnet_aux（可选安装 ComfyUI-AnimePose）")
    try:
        result = cls().estimate_pose(
            video_frames, detect_hand="enable", detect_body="enable", detect_face="enable",
            resolution=int(resolution), bbox_detector="yolox_l.onnx",
            pose_estimator="dw-ll_ucoco_384.onnx", scale_stick_for_xinsr_cn="disable")
    except TypeError:
        result = cls().estimate_pose(video_frames, resolution=int(resolution))
    return _control_image_result(result), type(cls()).__name__


def _run_depth_preprocessor(video_frames, resolution, mappings):
    # Prefer a temporal depth implementation when installed; otherwise use
    # the widely available Depth Anything V2 node.
    cls = next((mappings.get(name) for name in ("VideoDepthAnythingPreprocessor", "VideoDepthAnything")
                if mappings.get(name) is not None), None)
    if cls is not None:
        try:
            result = cls().execute(video_frames, resolution=int(resolution))
            return _control_image_result(result), type(cls()).__name__
        except (TypeError, RuntimeError, ValueError) as exc:
            LOG.warning("H3 Auto Director: Video Depth Anything 处理失败，回退 Depth Anything V2：%s", exc)
    cls = mappings.get("DepthAnythingV2Preprocessor")
    if cls is None:
        raise RuntimeError("未找到深度预处理器，请安装 comfyui_controlnet_aux 或 Video-Depth-Anything")
    result = cls().execute(video_frames, ckpt_name="depth_anything_v2_vitl.pth", resolution=int(resolution))
    return _control_image_result(result), type(cls()).__name__


def _persist_control_video(path, images):
    try:
        _write_segment_video(path, images.detach().float().cpu().contiguous(), None,
                             FPS, "mp4", "h264", "CPU", "最高质量")
        return bool(path.is_file() and path.stat().st_size > 0)
    except Exception as exc:
        # A .pt cache is still sufficient for an in-process/external runner;
        # missing ffmpeg should not abort H3 sampling.
        LOG.warning("H3 Auto Director: 控制视频 MP4 保存失败（已保留 PT 缓存）：%s", exc)
        return False


class H3AutoDirectorControlPreprocess:
    """Preprocess both pose and depth controls for one or all transfer segments.

    The output config is consumed by the dual sampler as conditioning metadata.
    Native H3 builds that do not expose the VideoX-Fun Union transformer safely
    ignore that metadata, while a compatible external backend can consume the
    persisted control-video paths and independent weights.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "control_mode": (list(CONTROL_MODES), {"default": "姿态+深度"}),
            "resolution": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
            "enabled": ("BOOLEAN", {"default": True, "label_on": "启用预处理", "label_off": "关闭预处理",
                                       "tooltip": "默认启用姿态/深度预处理。关闭时只对齐并转发原视频帧。"}),
            "save_preprocessed": ("BOOLEAN", {"default": True, "label_on": "保存预处理视频", "label_off": "不保存预处理视频",
                                                "tooltip": "默认保存到项目 control/segment_XXXX；同时保留 PT 缓存供后端读取。"}),
            "preprocess_all_segments": ("BOOLEAN", {"default": False, "label_on": "一次性预处理全部片段", "label_off": "仅当前片段",
                                                       "tooltip": "开启后按动作迁移计划完整预处理所有视频片段；适合开始采样前一次性生成控制缓存。"}),
            "pose_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "depth_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        }, "optional": {
            "video_frames": ("IMAGE", {"tooltip": "可选。连接完整参考视频 IMAGE；节点会按片段编号自动切分。未连接时从计划读取对应片段。"}),
            "plan": ("H3_AUTO_PLAN", {"tooltip": "连接动作迁移项目计划，自动读取并缓存对应参考视频片段。"}),
            "segment_index": ("INT", {"default": 1, "min": 1, "max": 9999,
                                         "tooltip": "动作迁移计划中的生成片段编号；应连接 H3 片段节点输出。"}),
            "source_video_path": ("STRING", {"default": "", "multiline": False,
                                               "tooltip": "可选。input 目录内的视频相对路径；优先级低于 video_frames 和 plan。"}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "H3_CONTROL_CONFIG", "IMAGE", "STRING")
    RETURN_NAMES = ("姿态控制视频", "深度控制视频", "控制配置", "控制预览", "预处理信息")
    FUNCTION = "preprocess"
    CATEGORY = "H3 自动导演/动作迁移/ControlNet"

    def preprocess(self, control_mode="姿态+深度", resolution=768, enabled=True,
                   save_preprocessed=True, preprocess_all_segments=False,
                   pose_weight=1.0, depth_weight=1.0, video_frames=None, plan=None,
                   segment_index=1, source_video_path="", **_legacy):
        mode = str(control_mode or "姿态+深度")
        # Retain old graph compatibility without exposing the retired style
        # widget.  The former control_type determines the new mode when used.
        if mode not in CONTROL_MODES:
            old_type = str(_legacy.get("control_type", mode))
            mode = "姿态" if old_type.startswith("姿态") else "深度" if old_type.startswith("深度") else "关闭"
        need_pose, need_depth = _control_mode_has_pose(mode), _control_mode_has_depth(mode)
        connected_frames = video_frames is not None
        if (video_frames is None and not bool(enabled) and not preprocess_all_segments
                and plan is None and not str(source_video_path or "").strip()):
            video_frames = torch.zeros((5, 16, 16, 3), dtype=torch.float32)
        if video_frames is None and isinstance(plan, dict):
            video_frames = _load_transfer_source_frames(plan, segment_index)
        if video_frames is None:
            source = str(source_video_path or "").strip().strip('"')
            if not source:
                raise ValueError("请连接视频帧、动作迁移项目计划，或填写 input 目录内的视频路径")
            try:
                video_frames, _ = _load_reference_video(source)
            except Exception as exc:
                if av is None:
                    raise RuntimeError("无法读取控制视频；请安装 VideoHelperSuite 或 PyAV") from exc
                video_frames = _load_video_frames_av((_input_root() / _reference_name(source)).resolve())
        if not torch.is_tensor(video_frames) or video_frames.ndim != 4:
            raise ValueError("控制视频输入必须是 [帧,高,宽,RGB] IMAGE")
        if int(video_frames.shape[0]) < 5:
            raise ValueError("控制视频至少需要 5 帧（约 0.2 秒）")
        full_frames = video_frames.float().clamp(0, 1).contiguous()
        indices = [max(1, int(segment_index))]
        if bool(preprocess_all_segments) and isinstance(plan, dict) and str(plan.get("mode", "")) == "video_transfer":
            indices = list(range(1, len(plan.get("segments", [])) + 1))
        mappings = _control_preprocessor_mappings()
        current_pose = current_depth = None
        records = {}
        for index in indices:
            if connected_frames:
                segment_frames = _control_frames_for_segment(plan, index, full_frames)
                if (segment_frames is full_frames and plan is not None
                        and str(plan.get("mode", "")) == "video_transfer"
                        and index != int(segment_index)):
                    # A connected batch that is already a segment cannot be
                    # reused for another window; load that plan window on demand.
                    segment_frames = _load_transfer_source_frames(plan, index)
            else:
                segment_frames = full_frames if index == int(segment_index) and not preprocess_all_segments \
                    else _load_transfer_source_frames(plan, index)
            if not torch.is_tensor(segment_frames) or segment_frames.ndim != 4:
                raise ValueError("控制视频片段必须是 [帧,高,宽,RGB] IMAGE")
            cache_pose, video_pose = _control_cache_paths(plan, index, "pose")
            cache_depth, video_depth = _control_cache_paths(plan, index, "depth")
            pose = _load_control_tensor(cache_pose) if need_pose else None
            depth = _load_control_tensor(cache_depth) if need_depth else None
            preprocessors = {}
            if bool(enabled):
                if need_pose and pose is None:
                    pose, preprocessors["pose"] = _run_pose_preprocessor(segment_frames, resolution, mappings)
                    pose, _ = _align_control_frames(pose)
                if need_depth and depth is None:
                    depth, preprocessors["depth"] = _run_depth_preprocessor(segment_frames, resolution, mappings)
                    depth, _ = _align_control_frames(depth)
            if not bool(enabled):
                passthrough, _ = _align_control_frames(segment_frames)
                pose = passthrough if need_pose else None
                depth = passthrough if need_depth else None
            if pose is not None:
                _save_control_tensor(cache_pose, pose)
                if bool(save_preprocessed) and video_pose is not None and not video_pose.is_file():
                    _persist_control_video(video_pose, pose)
            if depth is not None:
                _save_control_tensor(cache_depth, depth)
                if bool(save_preprocessed) and video_depth is not None and not video_depth.is_file():
                    _persist_control_video(video_depth, depth)
            records[str(index)] = {
                "pose_path": str(video_pose) if pose is not None and video_pose is not None else "",
                "depth_path": str(video_depth) if depth is not None and video_depth is not None else "",
                "pose_cache": str(cache_pose) if pose is not None and cache_pose is not None else "",
                "depth_cache": str(cache_depth) if depth is not None and cache_depth is not None else "",
                "frames": int(max(pose.shape[0] if pose is not None else 0, depth.shape[0] if depth is not None else 0)),
                "preprocessors": preprocessors,
            }
            if index == int(segment_index):
                current_pose, current_depth = pose, depth
        if current_pose is None and current_depth is None:
            fallback = _align_control_frames(full_frames)[0]
            current_pose = current_depth = fallback
        template = current_pose if current_pose is not None else current_depth
        blank = torch.zeros_like(template)
        payload = {
            "type": "h3_union_control",
            "enabled": bool(enabled) and mode != "关闭" and bool(need_pose or need_depth),
            "control_mode": mode,
            "pose_weight": max(0.0, min(2.0, float(pose_weight))) if need_pose else 0.0,
            "depth_weight": max(0.0, min(2.0, float(depth_weight))) if need_depth else 0.0,
            "segment_index": int(segment_index), "fps": FPS, "frame_grid": "17*n+5",
            "segments": records,
            "requires_videox_fun": True,
        }
        preview = current_pose if current_pose is not None else current_depth
        info = {"type": "h3_union_control", "mode": mode, "enabled": payload["enabled"],
                "pose_weight": payload["pose_weight"], "depth_weight": payload["depth_weight"],
                "current_segment": int(segment_index), "processed_segments": [int(x) for x in indices],
                "saved": bool(save_preprocessed), "records": records}
        LOG.info("H3 Auto Director: 控制预处理完成：模式=%s，姿态权重=%.3f，深度权重=%.3f，片段=%s",
                 mode, payload["pose_weight"], payload["depth_weight"], ",".join(map(str, indices)))
        return (current_pose if current_pose is not None else blank,
                current_depth if current_depth is not None else blank,
                payload, preview if preview is not None else blank,
                json.dumps(info, ensure_ascii=False))


def _load_video_frames_av(path):
    """Decode a video into ComfyUI IMAGE frames without VHS lazy audio state."""
    if av is None:
        raise RuntimeError("PyAV 未安装，无法直接读取控制视频")
    frames = []
    container = av.open(str(path))
    try:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            raise ValueError("控制视频没有视频流")
        try:
            source_fps = float(stream.average_rate or stream.base_rate or FPS)
        except (TypeError, ValueError):
            source_fps = FPS
        for frame in container.decode(stream):
            array = frame.to_rgb().to_ndarray()
            frames.append(torch.from_numpy(array).float().div(255.0))
    finally:
        container.close()
    if not frames:
        raise ValueError("控制视频没有可解码画面")
    if abs(source_fps - FPS) > 0.01 and len(frames) > 1:
        # Keep the source timeline's duration while selecting the nearest
        # frame for each 24-fps output timestamp.  This mirrors VHS's
        # ``force_rate=24`` behavior without creating a lazy audio mapping.
        target_count = max(1, int(round(len(frames) * FPS / source_fps)))
        indices = [min(len(frames) - 1, int(round(index * source_fps / FPS)))
                   for index in range(target_count)]
        frames = [frames[index] for index in indices]
    return torch.stack(frames, dim=0).contiguous()


class H3AutoDirectorControlConfig:
    """Describe how an external VideoX-Fun Union ControlNet run is configured."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "control_video": ("IMAGE",),
            "control_type": (["Pose", "Depth"], {"default": "Pose"}),
            "control_context_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "backend": (["VideoX-Fun Union（外部后端）", "仅记录配置（不启用控制）"], {"default": "VideoX-Fun Union（外部后端）"}),
            "controlnet_path": ("STRING", {"default": "", "multiline": False,
                                              "tooltip": "MiniMax-H3-Fun-Controlnet-Union.safetensors 的完整路径；仅外部 VideoX-Fun 后端使用"}),
        }}

    RETURN_TYPES = ("H3_CONTROL_CONFIG",)
    RETURN_NAMES = ("H3 Union 控制配置",)
    FUNCTION = "configure"
    CATEGORY = "H3 自动导演/动作迁移/ControlNet"

    def configure(self, control_video, control_type="Pose", control_context_scale=1.0,
                  backend="VideoX-Fun Union（外部后端）", controlnet_path=""):
        if not torch.is_tensor(control_video) or control_video.ndim != 4:
            raise ValueError("控制视频必须是 [帧,高,宽,RGB] IMAGE")
        frames, count = _align_control_frames(control_video.float().clamp(0, 1))
        path = str(controlnet_path or "").strip().strip('"')
        available = bool(path and Path(path).is_file())
        return ({
            "backend": str(backend), "control_type": str(control_type),
            "control_context_scale": max(0.0, min(2.0, float(control_context_scale))),
            "controlnet_path": path, "controlnet_available": available,
            "frames": int(count), "fps": FPS, "frame_grid": "17*n+5",
            "control_video": frames.contiguous(),
            "requires_videox_fun": str(backend).startswith("VideoX-Fun"),
        },)


class H3AutoDirectorControlExport:
    """Export pose/depth IMAGE frames for the external VideoX-Fun runner."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("H3_CONTROL_CONFIG",),
            "plan": ("H3_AUTO_PLAN",),
            "enabled": ("BOOLEAN", {"default": False, "label_on": "导出控制视频", "label_off": "不导出控制视频",
                                      "tooltip": "默认关闭；开启后才会调用 ffmpeg 将控制帧导出到项目 control 目录。"}),
            "output_name": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "控制视频文件名；留空按 pose/depth 自动命名。"}),
        }, "optional": {
            "video_codec": (["h264", "hevc"], {"default": "h264"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("控制视频路径", "导出状态")
    FUNCTION = "export"
    CATEGORY = "H3 自动导演/动作迁移/ControlNet"

    def export(self, config, plan, enabled=False, output_name="", video_codec="h264"):
        if not bool(enabled):
            return ("", "已关闭控制视频导出；原生 H3 动作迁移不使用 Union 控制")
        if not isinstance(config, dict) or not torch.is_tensor(config.get("control_video")):
            raise ValueError("控制配置中没有有效控制视频")
        if not isinstance(plan, dict) or not plan.get("project_dir"):
            raise ValueError("控制视频导出需要连接动作迁移项目计划")
        project_dir = Path(plan["project_dir"])
        control_dir = project_dir / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        control_type = "pose" if str(config.get("control_type", "Pose")).lower().startswith("pose") else "depth"
        name = _output_filename(output_name) if str(output_name or "").strip() else control_type
        path = control_dir / (name + ".mp4")
        images = config["control_video"].detach().float().clamp(0, 1).cpu().contiguous()
        _write_segment_video(path, images, None, FPS, "mp4", video_codec, "CPU", "最高质量")
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("控制视频导出失败：%s" % path)
        status = "已导出 %s 控制视频（%d 帧）；请交给 VideoX-Fun Union 专用管线执行" % (control_type, images.shape[0])
        LOG.info("H3 Auto Director: %s -> %s", status, path)
        return (str(path), status)


class H3AutoDirectorControlBackendCheck:
    """Fail early with an actionable message instead of silently ignoring Union control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": ("H3_CONTROL_CONFIG",)}}

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("后端可用", "检查信息")
    FUNCTION = "check"
    CATEGORY = "H3 自动导演/动作迁移/ControlNet"

    def check(self, config):
        try:
            import videox_fun  # noqa: F401
            package = True
        except Exception:
            package = False
        path_ok = bool(config.get("controlnet_available"))
        required = bool(config.get("requires_videox_fun"))
        ok = bool((not required) or (package and path_ok))
        if required and not ok:
            message = ("VideoX-Fun 后端不可用：请安装 VideoX-Fun，并将 "
                       "MiniMax-H3-Fun-Controlnet-Union.safetensors 路径填入控制配置节点。")
        elif required:
            message = "VideoX-Fun Union 控制后端与权重路径检查通过；该配置需交给专用 VideoX-Fun 采样节点执行。"
        else:
            message = "当前为仅记录配置模式，不会向原生 H3 采样注入 Union 控制。"
        return (ok, message)


def _h3_model_options(weight_dtype):
    options = {}
    if weight_dtype == "fp8_e4m3fn":
        options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        options["dtype"] = torch.float8_e4m3fn
        options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        options["dtype"] = torch.float8_e5m2
    return options


def _h3_hybrid_source(path):
    if st_safe_open is None:
        raise RuntimeError("当前 ComfyUI 环境缺少 safetensors，无法加载 H3 混合模型")
    if not Path(path).is_file():
        raise FileNotFoundError(f"找不到 H3 模型文件：{path}")
    if comfy.memory_management.aimdo_enabled:
        source, metadata = comfy.utils.load_torch_file(str(path), return_metadata=True)
        return source, set(source), metadata or {}
    source = st_safe_open(str(path), framework="pt", device="cpu")
    return source, set(source.keys()), source.metadata() or {}


def _h3_quant_parent(key):
    for suffix in (".comfy_quant", ".weight_scale", ".weight_scale_2", ".input_scale", ".pre_quant_scale"):
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return None


def _h3_hybrid_overlay_key(key, block_start=25, block_end=49):
    match = re.match(r"^blocks\.(\d+)\.adaln_proj\.linear\.(?:weight|bias)$", key)
    return match is not None and int(block_start) <= int(match.group(1)) <= int(block_end)


def _build_h3_hybrid_state_dict(base_path, overlay_path, block_start=25, block_end=49):
    """Merge only Ref2VA AdaLN blocks into FL2VA without writing a checkpoint.

    The int8 convrot files share their key layout and quantization sidecars. A
    sidecar is always selected from the same checkpoint as its parent weight,
    which prevents an invalid mixed quantization group.
    """
    base_source, base_keys, metadata = _h3_hybrid_source(base_path)
    overlay_source, overlay_keys, _ = _h3_hybrid_source(overlay_path)
    only_base = base_keys - overlay_keys
    only_overlay = overlay_keys - base_keys
    invalid_base = sorted(key for key in only_base if _h3_quant_parent(key) is None)
    invalid_overlay = sorted(key for key in only_overlay if _h3_quant_parent(key) is None)
    if invalid_base or invalid_overlay:
        raise RuntimeError(
            "FL2VA 与 Ref2VA 的模型键集合不兼容，无法混合："
            f" base-only={invalid_base[:5]} overlay-only={invalid_overlay[:5]}"
        )

    keys = set(base_keys)
    for key in only_overlay:
        parent = _h3_quant_parent(key)
        if parent and parent + ".weight" in base_keys and _h3_hybrid_overlay_key(parent + ".weight", block_start, block_end):
            keys.add(key)

    merged = {}
    for key in sorted(keys):
        take_overlay = _h3_hybrid_overlay_key(key, block_start, block_end)
        if not take_overlay:
            parent = _h3_quant_parent(key)
            take_overlay = bool(parent and _h3_hybrid_overlay_key(parent + ".weight", block_start, block_end))
        source, source_keys = (overlay_source, overlay_keys) if take_overlay else (base_source, base_keys)
        if key not in source_keys:
            raise RuntimeError(f"H3 混合模型的权重与量化附属数据不匹配：{key}")
        merged[key] = source[key] if isinstance(source, dict) else source.get_tensor(key)
    return merged, metadata


def _load_h3_hybrid_model(base_path, overlay_path, weight_dtype="default",
                          block_start=25, block_end=49, disable_dynamic=False):
    model_management.free_pins(1e32, evict_active=True, loaded=True)
    model_management.unload_all_models()
    # Do not pre-emptively reject hybrid loading based on a rough host-RAM
    # estimate.  The estimate does not account for ComfyUI's offload policy,
    # mapped safetensors, or available swap and could incorrectly block valid
    # configurations.  ComfyUI's model loader remains responsible for actual
    # allocation failures and VRAM/RAM management.
    state_dict, metadata = _build_h3_hybrid_state_dict(base_path, overlay_path, block_start, block_end)
    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options=_h3_model_options(weight_dtype),
        metadata=metadata, disable_dynamic=disable_dynamic,
    )
    if model is None:
        raise RuntimeError("ComfyUI 无法识别合并后的 H3 模型，请确认输入是 FL2VA/Ref2VA 检查点")
    model.cached_patcher_init = (
        _load_h3_hybrid_model,
        (base_path, overlay_path, weight_dtype, block_start, block_end),
    )
    return model


class H3AutoDirectorHybridModelLoader:
    """Load Ref2VA normally or optionally use the FL2VA/Ref2VA hybrid."""

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("diffusion_models")
        ref_matches = [name for name in files if "ref2va" in name.lower()]
        fl_matches = [name for name in files if "fl2va" in name.lower()]
        ref_default = next((name for name in ref_matches if "pruned_int8_convrot" in name.lower()), None) or (ref_matches[0] if ref_matches else None)
        fl_default = next((name for name in fl_matches if "pruned_int8_convrot" in name.lower()), None) or (fl_matches[0] if fl_matches else None)
        ref_options = {"tooltip": "关闭混合时使用的多模态参考模型（Ref2VA）"}
        base_options = {"tooltip": "开启混合时使用的画面基础模型（FL2VA）"}
        if ref_default:
            ref_options["default"] = ref_default
        if fl_default:
            base_options["default"] = fl_default
        return {"required": {
            "unet_name": (files, ref_options),
            "base_model": (files, base_options),
            "enable_hybrid": ("BOOLEAN", {"default": False, "label_on": "启用 H3 混合模型", "label_off": "关闭（仅 Ref2VA）",
                                             "tooltip": "关闭时只加载 Ref2VA；开启时使用 FL2VA 并覆盖 Ref2VA 第 25-49 个 AdaLN 块。"}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "advanced": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("模型",)
    FUNCTION = "load"
    CATEGORY = "H3 自动导演/模型加载"

    def load(self, unet_name, base_model, enable_hybrid=False, weight_dtype="default"):
        if not bool(enable_hybrid):
            return nodes.UNETLoader().load_unet(unet_name, weight_dtype)
        base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)
        overlay_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        LOG.info("[H3AutoDirector] 启用混合模型：FL2VA=%s Ref2VA=%s blocks=25..49",
                 os.path.basename(base_path), os.path.basename(overlay_path))
        return (_load_h3_hybrid_model(base_path, overlay_path, weight_dtype),)


class H3AutoDirectorTransferModelLoader:
    """Load the four H3 assets used by the experimental motion-transfer graph.

    Keeping the asset selection in one node makes the transfer workflow easier
    to audit while preserving the standard MODEL/CLIP/VAE socket contracts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            "base_model": (folder_paths.get_filename_list("diffusion_models"),),
            "enable_hybrid": ("BOOLEAN", {"default": False, "label_on": "启用 H3 混合模型", "label_off": "关闭（仅 Ref2VA）",
                                             "tooltip": "关闭时只加载 Ref2VA；开启时使用 FL2VA 并覆盖 Ref2VA 第 25-49 个 AdaLN 块。"}),
            "clip_name": (folder_paths.get_filename_list("text_encoders"),),
            "video_vae_name": (folder_paths.get_filename_list("vae"),),
            "audio_vae_name": (folder_paths.get_filename_list("vae"),),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"advanced": True}),
            "clip_type": (["minimax"], {"advanced": True}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("模型", "文本编码器", "视频 VAE", "音频 VAE")
    FUNCTION = "load"
    CATEGORY = "H3 自动导演/视频迁移"

    def load(self, unet_name, base_model, enable_hybrid, clip_name, video_vae_name, audio_vae_name,
             weight_dtype="default", clip_type="minimax"):
        if bool(enable_hybrid):
            base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_model)
            overlay_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            LOG.info("[H3AutoDirector] 动作迁移启用混合模型：FL2VA=%s Ref2VA=%s blocks=25..49",
                     os.path.basename(base_path), os.path.basename(overlay_path))
            model = _load_h3_hybrid_model(base_path, overlay_path, weight_dtype)
        else:
            model = nodes.UNETLoader().load_unet(unet_name, weight_dtype)[0]
        clip = nodes.CLIPLoader().load_clip(clip_name, clip_type, "default")[0]
        video_vae = nodes.VAELoader().load_vae(video_vae_name)[0]
        audio_vae = nodes.VAELoader().load_vae(audio_vae_name)[0]
        _log_h3_audio_vae(audio_vae, audio_vae_name)
        return (model, clip, video_vae, audio_vae)


class H3AutoDirectorDualStageModelLoader:
    """Load independent first/second-pass H3 models with optional FL2VA/Ref2VA hybrid weights.

    LoRA deliberately lives outside this node.  Standard ComfyUI model-only
    LoRA and memory/attention patches can therefore be inserted per stage.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("diffusion_models")
        return {"required": {
            "stage1_model": (models, {"default": models[0] if models else "", "tooltip": "一采多模态参考模型（Ref2VA）。"}),
            "stage1_base_model": (models, {"default": models[0] if models else "", "tooltip": "一采混合时使用的画面基础模型（FL2VA）。"}),
            "stage1_enable_hybrid": ("BOOLEAN", {"default": False, "label_on": "启用 H3 混合模型", "label_off": "关闭（仅 Ref2VA）"}),
            "stage2_model": (models, {"default": models[0] if models else "", "tooltip": "二采多模态参考模型（Ref2VA）。"}),
            "stage2_base_model": (models, {"default": models[0] if models else "", "tooltip": "二采混合时使用的画面基础模型（FL2VA）。"}),
            "stage2_enable_hybrid": ("BOOLEAN", {"default": False, "label_on": "启用 H3 混合模型", "label_off": "关闭（仅 Ref2VA）"}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default", "advanced": True}),
        }}

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("一采模型", "二采模型")
    FUNCTION = "load"
    CATEGORY = "H3 自动导演/模型加载"

    _MODEL_CACHE = {}

    def load(self, stage1_model, stage1_base_model="", stage1_enable_hybrid=False,
             stage2_model=None, stage2_base_model="", stage2_enable_hybrid=False,
             weight_dtype="default",
             **_legacy_unused):
        if not stage2_model:
            # A missing second-stage socket is a supported configuration, not
            # an invalid model name.  Mirror the dual sampler's fallback here
            # so direct API callers and older workflows behave identically.
            stage2_model = stage1_model
            stage2_enable_hybrid = stage1_enable_hybrid
            stage2_base_model = stage1_base_model
        def load_one(ref_name, use_hybrid, base_name):
            key = (str(ref_name), bool(use_hybrid), str(base_name), str(weight_dtype))
            if key not in self._MODEL_CACHE:
                if bool(use_hybrid):
                    base_path = folder_paths.get_full_path_or_raise("diffusion_models", base_name)
                    overlay_path = folder_paths.get_full_path_or_raise("diffusion_models", ref_name)
                    LOG.info("[H3AutoDirector] 分阶段加载器启用 H3 混合模型：FL2VA=%s Ref2VA=%s blocks=25..49",
                             os.path.basename(base_path), os.path.basename(overlay_path))
                    self._MODEL_CACHE[key] = _load_h3_hybrid_model(base_path, overlay_path, weight_dtype)
                else:
                    self._MODEL_CACHE[key] = nodes.UNETLoader().load_unet(ref_name, weight_dtype)[0]
            return self._MODEL_CACHE[key]
        same_stage = (stage2_model == stage1_model
                      and bool(stage2_enable_hybrid) == bool(stage1_enable_hybrid)
                      and stage2_base_model == stage1_base_model)
        first = load_one(stage1_model, stage1_enable_hybrid, stage1_base_model)
        if same_stage:
            return (first, first)
        second = load_one(stage2_model, stage2_enable_hybrid, stage2_base_model)
        return (first, second)


class H3AutoDirectorSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("H3_AUTO_PLAN",),
            "segment_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            "context_length": ("INT", {"default": 22, "min": 5, "max": 39}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN", "BOOLEAN", "STRING", "INT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("提示词", "帧数", "使用视频上下文", "使用音频上下文", "参考素材", "片段序号", "音频策略", "片段节点ID", "上下文序号")
    FUNCTION = "resolve"
    CATEGORY = "H3 自动导演"

    def resolve(self, plan, segment_index, context_length, unique_id=None):
        context_index = int(segment_index)
        generation_index = context_index + 1
        # SaveSegment can use this runtime-only value to remove the same
        # context prefix reserved for the current generation. It is not
        # persisted into project.json, so changing the workflow widget remains
        # immediately effective on the next queued segment.
        plan["_runtime_context_length"] = int(context_length)
        seg = _segment(plan, generation_index)
        use_previous_ref = _use_previous_video_reference(plan, generation_index)
        use_video = (not use_previous_ref and _video_context_enabled(plan)
                     and bool(seg.get("continue_video", context_index > 0)) and context_index > 0)
        restart = bool(seg.get("audio_restart", False))
        # Previous-video reference mode only disables pixel/video context. Keep
        # the audio-context decision independent so an audio-only policy can
        # still be represented by the segment output.
        use_audio = (bool(plan.get("continuation_mode", True)) and bool(seg.get("continue_audio", True))
                     and not restart and context_index > 0)
        target = round(float(seg["duration"]) * FPS)
        # Guide rows anchor the beginning of the denoised timeline. Reserve
        # only the context window; SaveSegment removes that same window so the
        # predecessor tail is not duplicated at the join.
        context_run = _h3_context_run(context_length)
        physical = (_align_frames_nearest(target + context_run)
                    if use_video else _align_frames(target))
        refs = _segment_reference_specs(plan, generation_index)
        LOG.info(
            "H3 Auto Director: 第 %d 段解析：视频上下文=%s，音频上下文=%s，"
            "上下文序号=%d，上一段视频参考=%s，计划视频开关=%s，片段视频开关=%s",
            generation_index, "开启" if use_video else "关闭", "开启" if use_audio else "关闭",
            context_index, "开启" if use_previous_ref else "关闭",
            "开启" if _video_context_enabled(plan) else "关闭",
            "开启" if bool(seg.get("continue_video", context_index > 0)) else "关闭",
        )
        return (_previous_video_prompt(seg.get("prompt", ""), refs), physical, use_video, use_audio,
                json.dumps(refs, ensure_ascii=False), generation_index,
                "restart" if restart else "continue", str(unique_id or ""), context_index)


def _reference_name(value):
    if isinstance(value, dict):
        value = value.get("path") or value.get("name") or ""
    name = str(value or "").strip().strip('"').replace("\\", "/")
    if name.startswith("input/"):
        name = name[6:]
    if not name or name in {".", ".."} or ".." in Path(name).parts:
        raise ValueError("参考素材必须位于 ComfyUI/input 目录内")
    return name


def _load_reference_image(name):
    image, _ = nodes.LoadImage().load_image(_reference_name(name))
    return image


def _load_reference_video(name):
    clean = _reference_name(name)
    video_path = (_input_root() / clean).resolve()
    loader = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideoPath")
    if loader is None:
        loader = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideo")
    if loader is None:
        raise RuntimeError("需要安装 VideoHelperSuite 才能加载视频参考素材")
    result = loader().load_video(video=str(video_path), force_rate=24,
                                 custom_width=0, custom_height=0, frame_load_cap=0,
                                 skip_first_frames=0, select_every_nth=1)
    soundtrack = result[2]
    # VHS exposes audio as a lazy mapping.  A silent video therefore looks
    # valid until the mapping is touched, at which point ffmpeg raises
    # "Output file does not contain any stream".  Inspect the container before
    # retaining that lazy object and pass a real None soundtrack for silent
    # references.  This keeps video-only conditioning usable.
    has_audio = _reference_video_has_audio(video_path)
    if has_audio is False:
        soundtrack = None
    elif has_audio is True and soundtrack is not None:
        try:
            # Force VHS's lazy map once so a malformed audio stream is handled
            # here instead of aborting the later H3 conditioning step.
            soundtrack["waveform"]
        except Exception as exc:
            LOG.warning("H3 Auto Director: 参考视频音轨无法读取，已仅传递画面：%s", exc)
            soundtrack = None
    return result[0], soundtrack


def _reference_video_has_audio(path):
    """Return True/False when the container stream layout is readable."""
    if av is not None:
        container = None
        try:
            container = av.open(str(path))
            return bool(container.streams.audio)
        except Exception:
            pass
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
    # PyAV is optional in some ComfyUI environments.  ffmpeg itself is
    # sufficient for the stream-layout check and, unlike VHS's lazy audio
    # mapping, does not fail when a file simply has no audio stream.
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(path)],
                capture_output=True, text=True, timeout=20,
            )
            details = (result.stdout or "") + "\n" + (result.stderr or "")
            if re.search(r"Stream #\d+:\d+[^\n]*Audio:", details, re.IGNORECASE):
                return True
            if re.search(r"Stream #\d+:\d+[^\n]*Video:", details, re.IGNORECASE):
                return False
        except (OSError, subprocess.SubprocessError):
            pass
    # A probe failure is deliberately unknown rather than silent.  In that
    # case VHS may still provide a valid lazy soundtrack for the caller.
    return None


def _load_transfer_video_segment(ref):
    """Load one 24 fps reference-video window and pad its tail to H3's grid."""
    clean = _reference_name(ref.get("path") or ref.get("name"))
    loader = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideoPath")
    if loader is None:
        loader = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideo")
    if loader is None:
        raise RuntimeError("需要安装 VideoHelperSuite 才能加载视频迁移参考素材")
    start_frame = max(0, int(ref.get("start_frame", 0)))
    source_frames = max(5, int(ref.get("source_frames", ref.get("reference_frames", 5))))
    reference_frames = max(source_frames, int(ref.get("reference_frames", source_frames)))
    result = loader().load_video(
        video=str((_input_root() / clean).resolve()), force_rate=24,
        custom_width=0, custom_height=0, frame_load_cap=source_frames,
        skip_first_frames=start_frame, select_every_nth=1)
    frames = result[0]
    if not torch.is_tensor(frames) or frames.shape[0] < 1:
        raise ValueError("参考视频片段没有可用画面")
    if frames.shape[0] < reference_frames:
        tail = frames[-1:].repeat((reference_frames - frames.shape[0], 1, 1, 1))
        frames = torch.cat((frames, tail), dim=0)
    return frames[:reference_frames]


def _load_transfer_video_audio(ref):
    """Load the corresponding source-audio time range, when the user enables it."""
    if ref.get("video_audio_enabled", True) is False:
        return None
    loader = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadAudio")
    if loader is None:
        LOG.warning("VideoHelperSuite audio loader unavailable; reference-video audio is skipped")
        return None
    clean = _reference_name(ref.get("path") or ref.get("name"))
    start_seconds = max(0.0, float(ref.get("start_frame", 0)) / FPS)
    duration_seconds = max(0.1, float(ref.get("source_frames", 5)) / FPS)
    video_path = (_input_root() / clean).resolve()
    if _reference_video_has_audio(video_path) is False:
        return None
    try:
        result = loader().load_audio(
            audio_file=str(video_path), seek_seconds=start_seconds,
            duration=duration_seconds)
        return result[0]
    except Exception as exc:
        # Reference video audio is optional.  A missing/corrupt audio stream
        # must not prevent the video frames from reaching H3.
        LOG.warning("H3 Auto Director: 参考视频没有可用音轨，已仅传递画面：%s", exc)
        return None


def _load_reference_audio(name):
    clean = _reference_name(name)
    loader = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadAudio")
    if loader is not None:
        result = loader().load_audio(audio_file=str((_input_root() / clean).resolve()), seek_seconds=0, duration=0)
        return result[0]
    loader = nodes.NODE_CLASS_MAPPINGS.get("LoadAudio")
    if loader is None:
        raise RuntimeError("需要安装音频加载节点才能加载音频参考素材")
    return loader.load(clean)[0]


def _input_root():
    return Path(folder_paths.get_input_directory()).resolve()


def _resolve_reference_groups(refs, plan=None):
    """Load references while preserving video/audio pairing indexes."""
    _validate_reference_limits(refs, "每段参考素材")
    images, videos, video_audios, standalone_audios = [], [], [], []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        kind = str(ref.get("type", "image")).lower()
        name = ref.get("path") or ref.get("name")
        if not name and kind != "previous_segment_video":
            continue
        if kind == "image":
            images.append(_load_reference_image(name))
        elif kind == "video":
            frames, soundtrack = _load_reference_video(name)
            if ref.get("video_audio_enabled", True) is False:
                soundtrack = None
            videos.append(frames)
            video_audios.append(soundtrack)
        elif kind == "transfer_video_segment":
            videos.append(_load_transfer_video_segment(ref))
            video_audios.append(_load_transfer_video_audio(ref))
        elif kind == "previous_segment_video":
            if plan is None:
                raise ValueError("上片段视频参考需要连接项目计划，以定位上下文视频缓存")
            previous_index = int(ref.get("segment_index", 0))
            video_path, _ = _paths(plan, previous_index, for_context=True)
            if not video_path.is_file():
                raise FileNotFoundError("上片段视频参考缓存不存在：%s" % video_path)
            videos.append(_load_context_video(video_path))
            # Deliberately do not append a soundtrack: this mode passes only video frames.
            video_audios.append(None)
        elif kind == "audio":
            standalone_audios.append(_load_reference_audio(name))
        else:
            raise ValueError("未知参考素材类型: %s" % kind)
    if len(images) > MAX_REFERENCE_IMAGES or len(videos) > MAX_REFERENCE_VIDEOS:
        raise ValueError("参考素材数量超过 H3 上限：图片最多 9 个，视频最多 3 个")
    if len(video_audios) > MAX_REFERENCE_VIDEOS or len(standalone_audios) > MAX_REFERENCE_AUDIOS:
        raise ValueError("音频参考素材数量超过 H3 上限：独立音频最多 3 个，视频音频最多 3 个")
    return (
        {f"ref_image_{i}": value for i, value in enumerate(images)},
        {f"ref_video_{i}": value for i, value in enumerate(videos)},
        {f"ref_video_audio_{i}": value for i, value in enumerate(video_audios)},
        {f"ref_audio_{i}": value for i, value in enumerate(standalone_audios)},
    )


def _reference_insert_frame(ref):
    """Return a 24-fps guide position; 0/0 intentionally means no guide."""
    if not isinstance(ref, dict):
        return None
    try:
        seconds = max(0.0, float(ref.get("insert_seconds", 0) or 0))
        frames = max(0, int(ref.get("insert_frames", 0) or 0))
    except (TypeError, ValueError):
        raise ValueError("素材插入时间必须是非负秒数和帧数")
    if seconds == 0 and frames == 0:
        return None
    return max(0, int(round(seconds * FPS)) + frames)


def _apply_reference_insert_guides(conditioning, latent, refs, vae, audio_vae):
    """Add explicitly timed image/video/audio guides on top of Ref2VA refs.

    The ordinary reference path remains unchanged.  This compatibility layer
    only runs for references carrying a non-zero insert position and only on
    cores that expose MiniMaxH3AddGuide; old ComfyUI versions still receive
    the same multimodal reference conditioning without a timed guide.
    """
    timed = [(ref, _reference_insert_frame(ref)) for ref in (refs or [])]
    timed = [(ref, pos) for ref, pos in timed if pos is not None]
    if not timed:
        return conditioning
    if _H3AddGuide is None:
        LOG.warning("H3 Auto Director: 当前 ComfyUI 没有 MiniMaxH3AddGuide，插入时间仅保留为参考素材")
        return conditioning
    output = conditioning
    parts = _av_latent_parts(latent)
    total_frames = _h3_pixel_frames(parts[0].shape[2]) if parts is not None else 0
    for ref, frame_idx in timed:
        kind = str(ref.get("type", "image")).lower()
        name = ref.get("path") or ref.get("name")
        if not name:
            continue
        image = audio = None
        if kind == "image":
            image = _load_reference_image(name)
        elif kind == "video":
            image, soundtrack = _load_reference_video(name)
            # MiniMaxH3AddGuide requires an entire guide clip to fit after its
            # anchor. Keep as much of the source window as the generated
            # timeline permits; a very short remainder deliberately becomes a
            # one-frame guide instead of failing the complete project.
            remaining = max(0, total_frames - int(frame_idx))
            if image is not None and image.shape[0] > remaining:
                if remaining < 5:
                    image = image[:1]
                else:
                    guide_frames = 5 + 17 * ((remaining - 5) // 17)
                    image = image[:guide_frames]
            if ref.get("video_audio_enabled", True) is not False:
                audio = soundtrack
        elif kind == "audio":
            audio = _load_reference_audio(name)
        else:
            continue
        if total_frames and int(frame_idx) >= total_frames:
            LOG.warning("H3 Auto Director: 素材插入位置 %d 超出本段 %d 帧，已跳过", frame_idx, total_frames)
            continue
        result = _H3AddGuide.execute(output, latent, frame_idx, vae=vae,
                                     audio_vae=audio_vae, image=image, audio=audio)
        try:
            output = result[0]
        except (TypeError, IndexError, KeyError):
            output = getattr(result, "result", result)
    return output


class H3AutoDirectorReferenceResolver:
    """Resolve the selected per-segment files into typed Ref2VA inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "references_json": ("STRING", {"default": "[]", "multiline": True, "forceInput": True}),
        }}

    RETURN_TYPES = tuple(["IMAGE"] * MAX_REFERENCE_IMAGES + ["IMAGE"] * MAX_REFERENCE_VIDEOS +
                         ["AUDIO"] * MAX_REFERENCE_VIDEOS + ["AUDIO"] * MAX_REFERENCE_AUDIOS)
    RETURN_NAMES = tuple(
        [f"图片{i}" for i in range(MAX_REFERENCE_IMAGES)] +
        [f"视频{i}" for i in range(MAX_REFERENCE_VIDEOS)] +
        [f"视频音频{i}" for i in range(MAX_REFERENCE_VIDEOS)] +
        [f"独立音频{i}" for i in range(MAX_REFERENCE_AUDIOS)]
    )
    FUNCTION = "resolve"
    CATEGORY = "H3 自动导演"

    def resolve(self, references_json):
        try:
            refs = json.loads(references_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError("参考素材 JSON 无效: %s" % exc) from exc
        groups = _resolve_reference_groups(refs)
        images = list(groups[0].values())
        videos = list(groups[1].values())
        video_audios = list(groups[2].values())
        standalone_audios = list(groups[3].values())
        images += [None] * (MAX_REFERENCE_IMAGES - len(images))
        videos += [None] * (MAX_REFERENCE_VIDEOS - len(videos))
        video_audios += [None] * (MAX_REFERENCE_VIDEOS - len(video_audios))
        standalone_audios += [None] * (MAX_REFERENCE_AUDIOS - len(standalone_audios))
        return tuple(images + videos + video_audios + standalone_audios)


def _cache_segment_references(plan, generation_index):
    return _segment_reference_specs(plan, generation_index)


def _cache_frame_count(plan, generation_index, context_length):
    seg = _segment(plan, generation_index)
    target = round(float(seg["duration"]) * FPS)
    use_video = (not _use_previous_video_reference(plan, generation_index)
                 and _video_context_enabled(plan)
                 and bool(seg.get("continue_video", generation_index > 1))
                 and generation_index > 1)
    return (_align_frames_nearest(target + _h3_context_run(context_length))
            if use_video else _align_frames(target))


def _prompt_cache_key(plan, clip, vae, audio_vae, width, height, ref_image_size, context_length,
                      ref_short_edge=2048):
    # The seed belongs to RandomNoise/sampling downstream.  It is deliberately
    # absent here so changing the seed reuses the deterministic H3 conditioning.
    plan_data = {k: plan.get(k) for k in ("project_id", "global_reference_set", "global_assets", "segments", "continuation_mode")}
    mode = str(ref_image_size or "match").lower()
    width, height = _h3_canvas_dimensions(width, height)
    resolution = (int(width), int(height)) if mode not in {"manual", "max"} else None
    return (id(clip), id(vae), id(audio_vae), resolution, mode,
            int(context_length), _nearest_multiple(ref_short_edge),
            json.dumps(plan_data, ensure_ascii=False, sort_keys=True, default=str))


def _refresh_cached_conditioning_latent(value, width, height, length):
    """Replace the generation latent carried by a cached conditioning.

    Reference/text conditioning in ``manual`` and ``max`` sizing modes is
    intentionally reusable across canvas changes.  The tuple returned by the
    encoder also carries an empty AV latent, however, and that latent *does*
    depend on width/height/length.  It is therefore rebuilt on every run,
    rather than conditionally refreshing only after a shape mismatch.
    """
    if _minimax_h3 is None or not isinstance(value, (tuple, list)) or len(value) < 2:
        return value
    conditioning = value[0]
    expected_width, expected_height = _h3_canvas_dimensions(width, height)
    # Do this unconditionally.  The latent embedded in a prompt-cache entry
    # is only a serialization convenience; it must never become a persistent
    # sampling state.  Rebuilding it on every execution guarantees that a
    # changed resolution/length gets a fresh AV container and that the
    # downstream RandomNoise seed is applied to a clean latent every run.
    expected_latent = _minimax_h3._empty_av_latent(
        expected_width, expected_height, int(length)
    )[0]
    LOG.info(
        "H3 Auto Director: 缓存文本向量复用，但按当前设置重建 AV latent：%dx%d，长度=%d",
        expected_width, expected_height, int(length),
    )
    if isinstance(value, tuple):
        return (conditioning, expected_latent, *value[2:])
    return [conditioning, expected_latent, *value[2:]]


class H3AutoDirectorCachedReferenceToVideo:
    """Reference-to-video node with optional one-shot per-project conditioning cache."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("H3_AUTO_PLAN", {"forceInput": True}),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "audio_vae": ("VAE",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "forceInput": True}),
            "width": ("INT", {"forceInput": True}),
            "height": ("INT", {"forceInput": True}),
            "length": ("INT", {"forceInput": True}),
            "ref_image_size": (["match", "max"], {"default": "match"}),
            "context_length": ("INT", {"default": FRAME_CONTEXT_DEFAULT, "min": 5, "max": 39}),
            "segment_index": ("INT", {"default": 0, "min": 0, "forceInput": True}),
            # Kept at the original widget position so old workflows still
            # deserialize correctly. It is now the visible preset-mode switch;
            # ``ref_image_size`` is the preset selector shown beside it.
            "use_auto_ref_image_size": ("BOOLEAN", {"default": True,
                                      "label_on": "使用预设",
                                      "label_off": "关闭使用预设",
                                      "tooltip": "使用 match/max 预设参考尺寸；关闭后自动切换到手动设置。"}),
            "use_manual_ref_short_edge": ("BOOLEAN", {"default": False,
                                      "label_on": "使用手动设置",
                                      "label_off": "关闭使用手动设置",
                                      "tooltip": "按参考图最短边缩放图片参考；输入会自动对齐到最近的 32 倍数。与预设模式二选一。"}),
            "ref_short_edge": ("INT", {"default": 2048, "min": 32, "max": 8192, "step": 32,
                                      "tooltip": "图片参考的目标最短边；不会放大低于该尺寸的原图。"}),
        }, "optional": {
            # The segment node emits this JSON directly. Keeping it optional
            # preserves older workflows while making per-segment references an
            # explicit data flow instead of relying only on plan lookup.
            "references_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
        }}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "LATENT")
    FUNCTION = "encode"
    CATEGORY = "H3 自动导演"

    @staticmethod
    def _encode_one(clip, vae, audio_vae, prompt, width, height, length, ref_image_size, refs,
                    plan=None, use_manual_ref_short_edge=False, ref_short_edge=2048):
        if _H3ReferenceToVideo is None:
            raise RuntimeError("当前 ComfyUI 未提供 MiniMaxH3ReferenceToVideo 核心节点")
        width, height = _h3_canvas_dimensions(width, height)
        LOG.info("H3 Auto Director: 参考编码画布=%dx%d（%.3f MP），参考尺寸模式=%s",
                 width, height, width * height / 1_000_000,
                 "manual" if use_manual_ref_short_edge else ref_image_size)
        if bool(use_manual_ref_short_edge):
            prepared = H3AutoDirectorCachedReferenceToVideo._prepare_references(
                vae, audio_vae, width, height, length, "manual", refs, plan=plan,
                ref_short_edge=ref_short_edge)
            cond = H3AutoDirectorCachedReferenceToVideo._encode_prepared_prompt(
                clip, prompt, *prepared)
            cond = _apply_reference_insert_guides(cond, prepared[0], refs, vae, audio_vae)
            return cond, prepared[0]
        ref_groups = _resolve_reference_groups(refs, plan=plan)
        result = _H3ReferenceToVideo.execute(
            clip, vae, audio_vae, prompt, int(width), int(height), int(length), str(ref_image_size),
            ref_images=ref_groups[0], ref_videos=ref_groups[1],
            ref_video_audios=ref_groups[2], ref_audios=ref_groups[3])
        cond = _apply_reference_insert_guides(result[0], result[1], refs, vae, audio_vae)
        return cond, result[1]

    @staticmethod
    def _prepare_references(vae, audio_vae, width, height, length, ref_image_size, refs,
                            plan=None, ref_short_edge=2048):
        """Encode all Ref2VA assets before the batch text-encoder session."""
        if _H3ReferenceToVideo is None or _minimax_h3 is None:
            raise RuntimeError("当前 ComfyUI 未提供 MiniMaxH3ReferenceToVideo 核心节点")
        width, height = _h3_canvas_dimensions(width, height)
        latent, frame_count = _minimax_h3._empty_av_latent(int(width), int(height), int(length))
        ref_groups = _resolve_reference_groups(refs, plan=plan)
        ref_items = []
        ref_blocks = []

        for image in ref_groups[0].values():
            if image is None:
                continue
            source_height, source_width = image.shape[1], image.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (source_width * source_height)))
            elif ref_image_size == "manual":
                target_short_edge = _nearest_multiple(ref_short_edge, _minimax_h3.CANVAS_MULTIPLE)
                scale = min(1.0, target_short_edge / min(source_width, source_height))
            else:
                scale = min(1.0, _minimax_h3.REF_IMAGE_SHORT_EDGE / min(source_width, source_height))
            target_width = max(_minimax_h3.CANVAS_MULTIPLE,
                               round(source_width * scale / _minimax_h3.CANVAS_MULTIPLE) * _minimax_h3.CANVAS_MULTIPLE)
            target_height = max(_minimax_h3.CANVAS_MULTIPLE,
                                round(source_height * scale / _minimax_h3.CANVAS_MULTIPLE) * _minimax_h3.CANVAS_MULTIPLE)
            resized = _minimax_h3._resize(image[:1], target_width, target_height, "disabled")
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": target_height // 16,
                               "latent_w": target_width // 16, "latent": vae.encode(resized)})

        video_audios = ref_groups[2]
        for name, video_frames in ref_groups[1].items():
            if video_frames is None:
                continue
            source_height, source_width = video_frames.shape[1], video_frames.shape[2]
            canvas_width, canvas_height = _minimax_h3.adapt_canvas(source_width, source_height)
            if source_width * source_height < canvas_width * canvas_height:
                canvas_width = max(_minimax_h3.CANVAS_MULTIPLE,
                                   round(source_width / _minimax_h3.CANVAS_MULTIPLE) * _minimax_h3.CANVAS_MULTIPLE)
                canvas_height = max(_minimax_h3.CANVAS_MULTIPLE,
                                    round(source_height / _minimax_h3.CANVAS_MULTIPLE) * _minimax_h3.CANVAS_MULTIPLE)
            frames = _minimax_h3._resize(video_frames, canvas_width, canvas_height, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            frame_total = frames.shape[0]
            if frame_total < 5:
                raise ValueError("MiniMax H3 参考视频至少需要 5 帧（24 fps 下约 0.2 秒）")
            while frame_total % 17 != 5:
                frame_total -= 1
            frames = frames[:frame_total]
            video_latent = vae.encode(frames)
            audio_latent, audio_length = (None, 0)
            soundtrack = video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            if soundtrack is not None:
                audio_latent, audio_length = _encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})
            sample_indices = list(range(0, frames.shape[0], _minimax_h3.FPS // 2))
            ref_items.append({"type": "video", "data": frames[sample_indices],
                              "timestamps": [index / 2.0 for index in range(len(sample_indices))]})
            ref_blocks.append({"kind": "video_audio" if audio_length else "video",
                               "latent_t": video_latent.shape[2], "latent_h": canvas_height // 16,
                               "latent_w": canvas_width // 16, "ref_audio_t": audio_length,
                               "latent": video_latent, "audio_latent": audio_latent})

        for audio in ref_groups[3].values():
            if audio is None:
                continue
            audio_latent, audio_length = _encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": audio_length,
                               "audio_latent": audio_latent})
        return latent, ref_items, ref_blocks

    @staticmethod
    def _encode_prepared_prompt(clip, prompt, latent, ref_items, ref_blocks):
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
        return conditioning, latent

    @staticmethod
    def _references_ready(plan, refs):
        """Previous-segment references become available only after their clip is saved."""
        for ref in refs:
            if not isinstance(ref, dict) or ref.get("type") != "previous_segment_video":
                continue
            previous_index = int(ref.get("segment_index", 0))
            video_path, _ = _paths(plan, previous_index, for_context=True)
            if not video_path.is_file():
                return False
        return True

    @classmethod
    def _build_cache(cls, plan, clip, vae, audio_vae, width, height, ref_image_size, context_length,
                     use_manual_ref_short_edge=False, ref_short_edge=2048):
        prepared = {}
        pending = []
        fingerprint_by_segment = {}
        segment_count = len(plan.get("segments", []))
        disk_enabled = _disk_cache_enabled(plan)
        model_identity = _cache_model_identity(clip, vae, audio_vae)
        disk_manifest = {}
        if disk_enabled:
            global_details = _prompt_disk_global_details(
                plan, width, height, ref_image_size, context_length,
                ref_short_edge, model_identity, use_manual_ref_short_edge)
            global_signature = _stable_digest(global_details)
            disk_manifest = _load_prompt_disk_manifest(plan)
            if disk_manifest.get("global_signature") != global_signature:
                disk_manifest = {"schema": PROMPT_DISK_CACHE_SCHEMA,
                                 "global_signature": global_signature,
                                 "global_details": global_details,
                                 "segments": {}}
                LOG.info("H3 Auto Director: 提示词磁盘缓存参数已变化，将按片段重新编码")
            else:
                LOG.info("H3 Auto Director: 已读取提示词向量磁盘缓存清单：%s",
                         _prompt_disk_paths(plan)[1])
        cache = {}
        LOG.info("H3 Auto Director: 正在预编码 %d 段参考素材，随后将连续缓存当前可用的文本向量", segment_count)
        for generation_index in range(1, len(plan.get("segments", [])) + 1):
            fingerprint = details = None
            if disk_enabled:
                fingerprint, details = _prompt_disk_fingerprint(
                    plan, generation_index, width, height,
                    "manual" if use_manual_ref_short_edge else ref_image_size,
                    context_length, ref_short_edge, model_identity)
                fingerprint_by_segment[generation_index] = (fingerprint, details)
                entry = (disk_manifest.get("segments") or {}).get(str(generation_index), {})
                cache_file = _prompt_disk_paths(plan)[0] / str(entry.get("file", ""))
                if entry.get("fingerprint") == fingerprint and cache_file.is_file():
                    try:
                        cache[generation_index] = _load_torch_cache(cache_file)
                        LOG.info("H3 Auto Director: 磁盘缓存命中 %d/%d（第 %d 段）",
                                 len(cache), segment_count, generation_index)
                        continue
                    except Exception as exc:
                        LOG.warning("H3 Auto Director: 第 %d 段磁盘向量无法读取，将重新编码：%s",
                                    generation_index, exc)
            length = _cache_frame_count(plan, generation_index, context_length)
            refs = _cache_segment_references(plan, generation_index)
            if not cls._references_ready(plan, refs):
                pending.append(generation_index)
                continue
            prepared[generation_index] = cls._prepare_references(
                vae, audio_vae, width, height, length,
                "manual" if use_manual_ref_short_edge else ref_image_size,
                refs, plan=plan, ref_short_edge=ref_short_edge)

        LOG.info("H3 Auto Director: 参考素材预编码完成，开始连续缓存 %d 段文本向量", len(prepared))
        for generation_index, prepared_segment in prepared.items():
            seg = _segment(plan, generation_index)
            refs = _cache_segment_references(plan, generation_index)
            encoded = cls._encode_prepared_prompt(
                clip, _previous_video_prompt(seg.get("prompt", ""), refs), *prepared_segment)
            cache[generation_index] = (
                _apply_reference_insert_guides(encoded[0], prepared_segment[0], refs, vae, audio_vae),
                encoded[1],
            )
            if disk_enabled:
                fingerprint, details = fingerprint_by_segment[generation_index]
                _save_prompt_disk_entry(plan, disk_manifest, generation_index,
                                        fingerprint, cache[generation_index], details)
            LOG.info(
                "H3 Auto Director: 已完成提示词向量缓存 %d/%d（第 %d 段）",
                len(cache), segment_count, generation_index,
            )
        if disk_enabled:
            # Persist the manifest even if this run only loaded existing entries;
            # it also records the active encoder/config signature for the next run.
            _atomic_json(_prompt_disk_paths(plan)[1], disk_manifest)
        if pending:
            LOG.info("H3 Auto Director: %d 段等待上片段视频生成，暂不卸载文本编码器；生成后按段补齐向量", len(pending))
        else:
            model_management.unload_model_and_clones(clip.patcher, unload_additional_models=False, all_devices=True)
            LOG.info("H3 Auto Director: 全部文本向量缓存完成，已卸载文本编码器")
        return cache

    @classmethod
    def _encode_current_with_disk_cache(cls, plan, clip, vae, audio_vae, prompt,
                                        width, height, length, ref_image_size,
                                        context_length, generation_index, refs,
                                        use_manual_ref_short_edge=False,
                                        ref_short_edge=2048):
        """Read/write only the requested segment when batch caching is off.

        The disk switch remains useful independently, but it must not turn a
        disabled one-shot cache into an eager all-segment text-encoder pass.
        """
        model_identity = _cache_model_identity(clip, vae, audio_vae)
        effective_ref_mode = "manual" if use_manual_ref_short_edge else ref_image_size
        fingerprint, details = _prompt_disk_fingerprint(
            plan, generation_index, width, height, effective_ref_mode,
            context_length, ref_short_edge, model_identity,
        )
        manifest = _load_prompt_disk_manifest(plan)
        global_details = _prompt_disk_global_details(
            plan, width, height, effective_ref_mode, context_length,
            ref_short_edge, model_identity,
            use_manual_ref_short_edge=use_manual_ref_short_edge,
        )
        global_signature = _stable_digest(global_details)
        if manifest.get("global_signature") != global_signature:
            manifest = {"schema": PROMPT_DISK_CACHE_SCHEMA,
                        "global_signature": global_signature,
                        "global_details": global_details,
                        "segments": {}}
        entry = (manifest.get("segments") or {}).get(str(generation_index), {})
        cache_root, _manifest_path = _prompt_disk_paths(plan)
        cache_file = cache_root / str(entry.get("file", ""))
        if entry.get("fingerprint") == fingerprint and cache_file.is_file():
            try:
                LOG.info("H3 Auto Director: 当前片段命中提示词磁盘缓存（第 %d 段）", generation_index)
                cached = _load_torch_cache(cache_file)
                return _refresh_cached_conditioning_latent(cached, width, height, length)
            except Exception as exc:
                LOG.warning("H3 Auto Director: 第 %d 段磁盘向量无法读取，将重新编码：%s",
                            generation_index, exc)
        conditioning = cls._encode_one(
            clip, vae, audio_vae,
            _previous_video_prompt(_segment(plan, generation_index).get("prompt", ""), refs),
            width, height, length, effective_ref_mode, refs, plan=plan,
            use_manual_ref_short_edge=use_manual_ref_short_edge,
            ref_short_edge=ref_short_edge,
        )
        _save_prompt_disk_entry(plan, manifest, generation_index,
                                fingerprint, conditioning, details)
        LOG.info("H3 Auto Director: 已保存当前片段提示词向量到硬盘（第 %d 段）", generation_index)
        return conditioning

    @classmethod
    def encode(cls, plan, clip, vae, audio_vae, prompt, width, height, length,
               ref_image_size="match", context_length=FRAME_CONTEXT_DEFAULT, segment_index=0,
               use_auto_ref_image_size=True, use_manual_ref_short_edge=False, ref_short_edge=2048,
               references_json=None):
        width, height = _h3_canvas_dimensions(width, height)
        LOG.info("H3 Auto Director: 编码请求画布=%dx%d（%.3f MP），帧数=%d",
                 width, height, width * height / 1_000_000, int(length))
        # The original Auto Director workflow wires output 8 (the 0-based
        # context index) into this node.  TTS and Video Transfer workflows
        # wire output 5 (the 1-based target segment number).  Keep both public
        # workflow contracts intact instead of forcing old projects to rewire.
        # This distinction is unambiguous from the persisted plan mode.
        raw_segment_index = int(segment_index)
        if str(plan.get("mode", "")) in {"tts", "video_transfer"}:
            generation_index = max(1, raw_segment_index)
        else:
            generation_index = raw_segment_index + 1
        refs = None
        if references_json is not None and str(references_json).strip():
            try:
                refs = json.loads(references_json)
            except json.JSONDecodeError as exc:
                raise ValueError("参考素材 JSON 无效: %s" % exc) from exc
            _validate_reference_limits(refs, "当前片段参考素材")
            if bool(plan.get("global_reference_set", True)) and generation_index != 1:
                refs = [
                    dict(ref, insert_seconds=0.0, insert_frames=0)
                    if isinstance(ref, dict) else ref
                    for ref in refs
                ]
        # The UI keeps the two switches mutually exclusive. If an old or
        # hand-edited workflow contains both values, preset mode wins; if both
        # are off, preset mode is the deterministic fallback.
        manual_enabled = bool(use_manual_ref_short_edge) and not bool(use_auto_ref_image_size)
        cache_all_enabled = _prompt_cache_all_enabled(plan)
        disk_enabled = _disk_cache_enabled(plan)
        if not cache_all_enabled:
            effective_refs = _cache_segment_references(plan, generation_index) if refs is None else refs
            if disk_enabled:
                return cls._encode_current_with_disk_cache(
                    plan, clip, vae, audio_vae, prompt, width, height, length,
                    ref_image_size, context_length, generation_index, effective_refs,
                    use_manual_ref_short_edge=manual_enabled,
                    ref_short_edge=ref_short_edge,
                )
            return cls._encode_one(clip, vae, audio_vae, prompt, width, height, length,
                                   ref_image_size, effective_refs,
                                   plan=plan, use_manual_ref_short_edge=manual_enabled,
                                   ref_short_edge=ref_short_edge)
        effective_ref_mode = "manual" if manual_enabled else ref_image_size
        key = _prompt_cache_key(plan, clip, vae, audio_vae, width, height,
                                effective_ref_mode, context_length, ref_short_edge)
        cache = _PROMPT_CONDITIONING_CACHE.get(key)
        if cache is None:
            cache = cls._build_cache(plan, clip, vae, audio_vae, width, height, ref_image_size, context_length,
                                     use_manual_ref_short_edge=manual_enabled,
                                     ref_short_edge=ref_short_edge)
            _PROMPT_CONDITIONING_CACHE[key] = cache
            _PROMPT_CONDITIONING_CACHE.move_to_end(key)
            while len(_PROMPT_CONDITIONING_CACHE) > PROMPT_CACHE_MAX_PROJECTS:
                _PROMPT_CONDITIONING_CACHE.popitem(last=False)
        else:
            _PROMPT_CONDITIONING_CACHE.move_to_end(key)
        if generation_index not in cache:
            if generation_index > len(plan.get("segments", [])):
                raise ValueError("segment_index %d 对应的下一段不存在" % int(segment_index))
            seg = _segment(plan, generation_index)
            effective_refs = _cache_segment_references(plan, generation_index) if refs is None else refs
            if not cls._references_ready(plan, effective_refs):
                raise FileNotFoundError("第 %d 段的上片段视频参考尚未生成，无法编码提示词向量" % generation_index)
            cache[generation_index] = cls._encode_one(
                clip, vae, audio_vae, _previous_video_prompt(seg.get("prompt", ""), effective_refs),
                width, height, length, effective_ref_mode, effective_refs, plan=plan,
                use_manual_ref_short_edge=manual_enabled,
                ref_short_edge=ref_short_edge)
            _PROMPT_CONDITIONING_CACHE[key] = cache
            if _disk_cache_enabled(plan):
                model_identity = _cache_model_identity(clip, vae, audio_vae)
                fingerprint, details = _prompt_disk_fingerprint(
                    plan, generation_index, width, height, effective_ref_mode,
                    context_length, ref_short_edge, model_identity)
                manifest = _load_prompt_disk_manifest(plan)
                if not manifest:
                    global_details = _prompt_disk_global_details(
                        plan, width, height, effective_ref_mode, context_length,
                        ref_short_edge, model_identity,
                        use_manual_ref_short_edge=manual_enabled)
                    manifest = {"schema": PROMPT_DISK_CACHE_SCHEMA,
                                "global_signature": _stable_digest(global_details),
                                "global_details": global_details,
                                "segments": {}}
                _save_prompt_disk_entry(plan, manifest, generation_index,
                                        fingerprint, cache[generation_index], details)
            LOG.info(
                "H3 Auto Director: 已完成提示词向量缓存 %d/%d（第 %d 段，延迟参考素材就绪）",
                len(cache), len(plan.get("segments", [])), generation_index,
            )
            if len(cache) >= len(plan.get("segments", [])):
                model_management.unload_model_and_clones(clip.patcher, unload_additional_models=False, all_devices=True)
                LOG.info("H3 Auto Director: 延迟参考素材就绪，全部文本向量缓存完成，已卸载文本编码器")
        cache[generation_index] = _refresh_cached_conditioning_latent(
            cache[generation_index], width, height, length
        )
        return cache[generation_index]


def _transfer_segment_ref(plan, generation_index):
    for ref in _segment(plan, generation_index).get("references", []):
        if isinstance(ref, dict) and ref.get("type") == "transfer_video_segment":
            return ref
    raise ValueError("视频迁移计划缺少当前片段的参考视频窗口")


class H3AutoDirectorTransferDecode:
    """Decode video and conditionally keep H3 audio, source audio, or no audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("H3_AUTO_PLAN",), "segment_index": ("INT", {"default": 0, "min": 0, "forceInput": True}),
            "samples": ("LATENT",), "video_vae": ("VAE",), "audio_vae": ("VAE",),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("画面", "音频")
    FUNCTION = "decode"
    CATEGORY = "H3 自动导演/视频迁移"

    def decode(self, plan, segment_index, samples, video_vae, audio_vae):
        parts = _av_latent_parts(samples)
        if parts is None:
            raise ValueError("视频迁移解码仅支持 MiniMax H3 的联合 AV latent")
        if isinstance(plan, dict) and bool(plan.get("decode_after_all_segments", False)):
            LOG.info("H3 Auto Director: 统一解码模式跳过动作迁移片段级视频/音频解码")
            return (torch.empty((0, 1, 1, 3), dtype=torch.float32), None)
        video, audio = parts
        images = _decode_h3_video(video_vae, video)
        source_audio = str(plan.get("final_audio_source", "H3 生成音频")) == "参考视频音频"
        skip = bool(plan.get("skip_h3_audio_decode", False))
        if source_audio or skip:
            if source_audio:
                return (images, _load_transfer_video_audio(_transfer_segment_ref(plan, int(segment_index) + 1)))
            return (images, None)
        waveform, sample_rate = _decode_h3_audio(audio_vae, audio)
        return (images, {"waveform": waveform, "sample_rate": sample_rate})


class H3AutoDirectorAVDecode:
    """Decode the final H3 joint AV latent for the standard director plan."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "video_vae": ("VAE",), "audio_vae": ("VAE",),
        }, "optional": {"plan": ("H3_AUTO_PLAN",)}}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("画面", "音频")
    FUNCTION = "decode"
    CATEGORY = "H3 自动导演/解码"

    def decode(self, samples, video_vae, audio_vae, plan=None):
        parts = _av_latent_parts(samples)
        if parts is None:
            raise ValueError("H3 AV 解码需要联合视频/音频 latent")
        if isinstance(plan, dict) and bool(plan.get("decode_after_all_segments", False)):
            # SaveSegment persists the latent; Controller decodes one segment
            # at a time during final assembly. Keep the output contract so old
            # links remain valid without allocating a decoded clip here.
            return (torch.empty((0, 1, 1, 3), dtype=torch.float32), None)
        images = _decode_h3_video(video_vae, parts[0])
        waveform, sample_rate = _decode_h3_audio(audio_vae, parts[1])
        # Native H3 Guide has no prepended context frames. Its audio latent
        # grid can round up by a fraction of a frame, so trim only that tail
        # here and keep every decoded video frame.
        expected_samples = int(round(int(images.shape[0]) / FPS * sample_rate))
        if waveform.shape[-1] > expected_samples:
            waveform = waveform[..., :expected_samples]
        return (images, {"waveform": waveform, "sample_rate": sample_rate})


class H3AutoDirectorLoadSavedAVLatent:
    """Load an AV latent cache written by H3 Auto Director without a plan.

    This deliberately accepts only the paired ``video`` / ``audio`` tensors
    written by Save Segment.  Loading it through a normal generic latent node
    loses the two-stream H3 contract and makes audio decode silently fail.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent_path": ("STRING", {"default": "", "multiline": False,
                             "placeholder": "D:/.../output/h3_project/项目/cache/segment_0001.safetensors"}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("H3 AV 潜空间", "已加载文件")
    FUNCTION = "load"
    CATEGORY = "H3 自动导演/解码"

    @staticmethod
    def _path(value):
        return Path(str(value or "").strip().strip('"')).expanduser()

    @classmethod
    def VALIDATE_INPUTS(cls, latent_path):
        path = cls._path(latent_path)
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            return "潜空间文件必须是存在的 .safetensors：%s" % path
        return True

    @classmethod
    def IS_CHANGED(cls, latent_path):
        path = cls._path(latent_path)
        try:
            stat = path.stat()
            return "%s:%d:%d" % (path.resolve(), stat.st_size, stat.st_mtime_ns)
        except OSError:
            return str(path)

    def load(self, latent_path):
        path = self._path(latent_path)
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            raise FileNotFoundError("潜空间文件必须是存在的 .safetensors：%s" % path)
        latent = _load_av_latent(path)
        video, audio = _av_latent_parts(latent) or (None, None)
        if video is None or audio is None:
            raise ValueError("不是 H3 自动导演保存的联合 AV 潜空间：%s" % path)
        LOG.info("H3 Auto Director: 已加载保存的 AV 潜空间：%s（视频=%s，音频=%s）",
                 path, tuple(video.shape), tuple(audio.shape))
        return (latent, str(path))


class H3AutoDirectorDecodeSaveVideo:
    """Decode every saved AV latent in a project directory and assemble video.

    The node deliberately keeps only one decoded segment in memory.  It accepts
    either a project directory (``.../h3_project/<name>``), its ``cache``
    directory, or ``cache_stage1``.  Stage-one caches are useful for diagnosis;
    normal output should point at ``cache`` so the final second-pass latents are
    decoded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent_directory": ("STRING", {"default": "", "multiline": False,
                "placeholder": ".../output/h3_project/项目 或其 cache/ 目录"}),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "output_intermediate": ("BOOLEAN", {"default": True,
                "label_on": "输出中间片段", "label_off": "不输出中间片段"}),
            "intermediate_filename": ("STRING", {"default": "H3",
                "tooltip": "中间片段文件名前缀；留空使用 H3"}),
            "final_filename": ("STRING", {"default": "H3",
                "tooltip": "最终视频文件名；留空使用 H3"}),
            "auto_crop_frames": ("INT", {"default": 22, "min": 0, "max": 4096,
                "tooltip": "从第 2 段开始裁剪的上下文帧数；0 表示不裁剪"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("最终视频", "保存信息")
    FUNCTION = "decode_save"
    CATEGORY = "H3 自动导演/解码"
    OUTPUT_NODE = True

    @staticmethod
    def _project_and_cache(value):
        raw = str(value or "").strip().strip('"')
        if not raw:
            raise ValueError("请输入项目目录或 cache/ 目录")
        path = Path(raw).expanduser()
        if path.is_file():
            path = path.parent
        path = path.resolve()
        if path.name.lower() in {"cache", "cache_stage1", "latents"}:
            return path.parent, path
        if (path / "cache").is_dir():
            return path, path / "cache"
        if (path / "cache_stage1").is_dir():
            return path, path / "cache_stage1"
        # A direct directory containing safetensors is also accepted.  This
        # keeps old projects and manually copied caches resumable.
        if path.is_dir() and any(path.glob("*.safetensors")):
            return path.parent, path
        raise FileNotFoundError("无法识别潜空间目录（应为项目目录或 cache/）：%s" % path)

    @staticmethod
    def _entries(cache_dir):
        files = [p for p in cache_dir.glob("*.safetensors") if p.is_file()]
        def key(path):
            match = re.search(r"(?:_|-)(\d+)(?:\.safetensors)?$", path.name)
            return (int(match.group(1)) if match else 10**9, path.name.lower())
        return sorted(files, key=key)

    @staticmethod
    def _safe_name(value, default):
        name = str(value or "").strip().strip('"')
        name = Path(name).stem if name else default
        if not name or name in {".", ".."} or any(c in name for c in "\\/:*?<>|\n\r"):
            raise ValueError("文件名只能包含文件名，不能包含路径或特殊字符")
        return name

    def decode_save(self, latent_directory, video_vae, audio_vae,
                    output_intermediate=True, intermediate_filename="H3",
                    final_filename="H3", auto_crop_frames=22):
        project_dir, cache_dir = self._project_and_cache(latent_directory)
        entries = self._entries(cache_dir)
        if not entries:
            raise FileNotFoundError("目录中没有 .safetensors 潜空间：%s" % cache_dir)
        intermediate_stem = self._safe_name(intermediate_filename, "H3")
        final_stem = self._safe_name(final_filename, "H3")
        crop = max(0, int(auto_crop_frames or 0))
        clips_dir = project_dir / "clips"
        final_dir = project_dir / "final"
        temp_dir = project_dir / "json" / ".decode_segments"
        target_dir = clips_dir if bool(output_intermediate) else temp_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        list_path = project_dir / "json" / "decode_concat.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        sources = []
        decoded_count = 0
        try:
            for ordinal, latent_path in enumerate(entries, 1):
                latent = _load_av_latent(latent_path)
                parts = _av_latent_parts(latent)
                if parts is None:
                    raise ValueError("不是联合 AV latent：%s" % latent_path)
                images = _decode_h3_video(video_vae, parts[0])
                waveform, sample_rate = _decode_h3_audio(audio_vae, parts[1])
                audio = {"waveform": waveform, "sample_rate": sample_rate}
                if ordinal >= 2 and crop > 0:
                    if crop >= int(images.shape[0]):
                        raise ValueError("第 %d 段裁剪 %d 帧后没有剩余画面" % (ordinal, crop))
                    images, audio = _trim_context_prefix(images, audio, crop, FPS)
                clip_path = target_dir / ("%s_%05d.mp4" % (intermediate_stem, ordinal))
                _write_segment_video(clip_path, images, audio, FPS, "mp4", "h264", "CPU", "最高质量")
                sources.append(clip_path)
                decoded_count += 1
                LOG.info("H3 Auto Director: 目录解码第 %d/%d 段（裁剪=%d）: %s", ordinal, len(entries), crop if ordinal >= 2 else 0, latent_path)
                del latent, parts, images, waveform, audio
                _release_video_memory()
            list_path.write_text("\n".join("file '%s'" % str(p).replace("'", "'\\''") for p in sources) + "\n", encoding="utf-8")
            ffmpeg = _find_ffmpeg()
            if not ffmpeg:
                raise RuntimeError("未找到 ffmpeg，无法拼接解码视频")
            final_path = final_dir / (final_stem + ".mp4")
            temp_final = final_path.with_name(final_path.stem + ".tmp.mp4")
            _encode_concat_with_fallback(ffmpeg, list_path, temp_final, "mp4", "h264", "CPU", "最高质量")
            if not temp_final.is_file() or temp_final.stat().st_size == 0:
                raise RuntimeError("ffmpeg 未生成有效最终视频")
            os.replace(temp_final, final_path)
            info = "目录=%s，读取=%d 段，裁剪=%d 帧（从第2段起），中间片段=%s，最终=%s" % (cache_dir, decoded_count, crop, "开启" if output_intermediate else "关闭", final_path)
            LOG.info("H3 Auto Director: %s", info)
            return (str(final_path), info)
        finally:
            list_path.unlink(missing_ok=True)
            if not bool(output_intermediate):
                shutil.rmtree(temp_dir, ignore_errors=True)
            _release_video_memory()


class H3AutoDirectorAudioDecode:
    """Decode only H3 audio latent; the video latent is deliberately untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "audio_vae": ("VAE",),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "decode"
    CATEGORY = "H3 自动导演/TTS"

    def decode(self, samples, audio_vae):
        parts = _av_latent_parts(samples)
        if parts is None:
            raise ValueError("TTS 音频解码需要 MiniMax H3 联合 AV latent")
        waveform, sample_rate = _decode_h3_audio(audio_vae, parts[1])
        return ({"waveform": waveform, "sample_rate": sample_rate},)


class H3AutoDirectorSaveAudioSegment:
    """Save one TTS segment audio file while keeping the latent cache fixed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("H3_AUTO_PLAN",), "segment_index": ("INT", {"default": 1, "min": 1}),
            "latent": ("LATENT",), "audio": ("AUDIO",),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("已保存音频", "已保存潜变量")
    FUNCTION = "save"
    CATEGORY = "H3 自动导演/TTS"
    OUTPUT_NODE = True

    def save(self, plan, segment_index, latent, audio):
        runtime = _runtime_plan(plan)
        index = int(segment_index)
        filename = _segment(runtime, index).get("audio_filename", "")
        audio_path = _audio_path(runtime, index, filename, for_write=True)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        if audio is None:
            raise ValueError("TTS 音频为空，无法保存")
        if _write_wav(audio_path, audio) is None:
            raise ValueError("TTS 音频没有 waveform")
        parts = _av_latent_parts(latent)
        if parts is None or st_save is None:
            raise ValueError("Sampler output must be an H3 video/audio latent pair")
        _, latent_path = _paths(runtime, index, for_write=True)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        st_save({"video": parts[0].detach().cpu().contiguous(), "audio": parts[1].detach().cpu().contiguous()},
                str(latent_path), metadata={"format": "h3_auto_director_av_v1", "segment_index": str(index)})
        state_path = _state_path(runtime)
        state = _load_json(state_path, {"version": 3, "segments": {}})
        state.setdefault("segments", {})[str(index)] = {"status": "completed", "audio": str(audio_path)}
        state["last_completed"] = index
        _atomic_json(state_path, state)
        return (str(audio_path), str(latent_path))


class H3AutoDirectorContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("H3_AUTO_PLAN",), "segment_index": ("INT", {"default": 0, "min": 0})},
                "optional": {"context_stage": ("INT", {"default": 1, "min": 1, "max": 2,
                    "tooltip": "1=一采上下文，2=二采最终上下文"})}}

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("上下文画面", "上下文潜变量")
    FUNCTION = "load"
    CATEGORY = "H3 自动导演"

    def load(self, plan, segment_index, context_stage=1):
        video_enabled = _video_context_enabled(plan)
        audio_enabled = bool(plan.get("continuation_mode", True))
        if int(segment_index) <= 0 or not (video_enabled or audio_enabled):
            LOG.info("H3 Auto Director: 上下文序号 %d 未启用视频/音频上下文，返回空上下文", int(segment_index))
            return (torch.zeros((1, 1, 1, 3), dtype=torch.float32), {"samples": [torch.zeros((1, 24, 2, 1, 1)), torch.zeros((1, 32, 2, 1))]})
        context_stage = 1 if int(context_stage) == 1 else 2
        requested_video_path, requested_latent_path = _paths(
            plan, int(segment_index), for_context=True, context_stage=context_stage
        )
        video_path, latent_path = requested_video_path, requested_latent_path
        if context_stage == 1:
            # Keep the first pass independent from the final second-pass
            # output.  Feeding a soft second-pass prefix into the next first
            # pass creates a feedback loop: the blur is then refined and
            # propagated through every subsequent segment.  The second pass
            # receives the explicit stage-two source below when its context
            # option is enabled.
            LOG.info(
                "H3 Auto Director: 一采上下文固定读取上一段一采源：视频=%s，latent=%s",
                video_path, latent_path,
            )
            if not latent_path.is_file():
                # Pre-isolation projects did not always save a separate
                # first-pass cache. Keep those projects resumable, but never
                # prefer this fallback when the dedicated cache exists.
                final_video_path, final_latent_path = _paths(
                    plan, int(segment_index), for_context=True, context_stage=2
                )
                if final_latent_path.is_file() and (not video_enabled or final_video_path.is_file()):
                    video_path, latent_path = final_video_path, final_latent_path
                    LOG.warning(
                        "H3 Auto Director: 一采缓存缺失，兼容回退到上一段最终二采上下文：%s",
                        latent_path,
                    )
            # The workflow-compatible frame Guide reads the saved final video
            # even when a dedicated stage-one preview was not connected. Keep
            # the stage-one latent above for latent consumers, but source its
            # decoded frames from the final context directory in that case.
            if video_enabled and not video_path.is_file():
                final_video_path, _ = _paths(
                    plan, int(segment_index), for_context=True, context_stage=2
                )
                if final_video_path.is_file():
                    video_path = final_video_path
                    LOG.info(
                        "H3 Auto Director: 一采帧 Guide 未找到独立一采视频，改用上一段最终视频：%s",
                        video_path,
                    )
        # Stage-one context intentionally stores the latent first.  Its video
        # preview is optional because latent-direct Motion Context does not
        # need a decoded frame stream; stage two keeps the normal video cache.
        deferred_decode = bool(plan.get("decode_after_all_segments", False))
        if not latent_path.exists() or (video_enabled and context_stage != 1 and not video_path.exists() and not deferred_decode):
            raise FileNotFoundError("Missing context cache for segment %d: %s / %s" % (int(segment_index), video_path, latent_path))
        LOG.info(
            "H3 Auto Director: 加载上下文序号 %d（阶段%d）：视频=%s，音频=%s，视频缓存=%s，latent缓存=%s",
            int(segment_index), context_stage, "开启" if video_enabled else "关闭",
            "开启" if audio_enabled else "关闭", video_path, latent_path,
        )
        frames = (_load_context_video(video_path) if video_enabled and video_path.exists() and not deferred_decode
                  else torch.zeros((1, 1, 1, 3), dtype=torch.float32))
        context_latent = _load_av_latent(latent_path)
        if bool(plan.get("decode_after_all_segments", False)):
            # Deferred decoding deliberately leaves no per-segment context
            # video on disk. Motion Context must use this durable AV latent.
            context_latent = dict(context_latent)
            context_latent["h3_deferred_decode"] = True
        # Keep the final stage-two source explicit for the second pass.  When
        # stage one already selected that source this is a duplicate by
        # design; the marker makes the routing unambiguous and keeps old
        # workflows (where stage-one cache is still the primary source)
        # compatible.
        if context_stage == 1:
            _, stage2_path = _paths(plan, int(segment_index), for_context=True, context_stage=2)
            if stage2_path.is_file():
                context_latent = dict(context_latent)
                context_latent["h3_stage2_context_latent"] = _load_av_latent(stage2_path)
        return (frames, context_latent)


class H3AutoDirectorResumeContext:
    """Explicit, opt-in loading of a user-selected latent/video pair."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enable_resume": ("BOOLEAN", {"default": False}),
            "latent_path": ("STRING", {"default": "", "multiline": False}),
            "video_path": ("STRING", {"default": "", "multiline": False}),
        }}

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("上下文画面", "上下文潜变量")
    FUNCTION = "load"
    CATEGORY = "H3 自动导演"

    def load(self, enable_resume, latent_path, video_path):
        placeholder_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        placeholder_latent = {"samples": [torch.zeros((1, 24, 2, 1, 1)), torch.zeros((1, 32, 2, 1))]}
        if not enable_resume:
            return (placeholder_image, placeholder_latent)
        latent = Path(str(latent_path).strip().strip('"')).expanduser()
        video = Path(str(video_path).strip().strip('"')).expanduser()
        if not latent.is_file() or latent.suffix.lower() != ".safetensors":
            raise FileNotFoundError("Resume latent must be an existing .safetensors file")
        if not video.is_file():
            raise FileNotFoundError("Resume video must be an existing video file")
        return (_load_context_video(video), _load_av_latent(latent))


class H3AutoDirectorMotionContext:
    """Adapter around the installed Motion Context node with audio reset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",), "vae": ("VAE",), "latent": ("LATENT",),
            "context_frames": ("IMAGE",), "use_video_context": ("BOOLEAN", {"default": False}),
            "use_audio_context": ("BOOLEAN", {"default": True}), "context_length": ("INT", {"default": 22, "min": 5, "max": 39}),
        }, "optional": {
            "context_latent": ("LATENT",),
            "use_video_latent": ("BOOLEAN", {"default": True, "tooltip": "优先直接使用缓存 AV latent 的视频尾部；尺寸不匹配时自动回退至画面编码。"}),
            "context_method": (["工作流视频帧 Guide", "缓存视频 latent 直取", "自动（latent 优先）"],
                                {"default": "工作流视频帧 Guide",
                                 "tooltip": "工作流视频帧 Guide：从磁盘视频解码尾帧后编码并通过 H3 AddGuide 注入；缓存视频 latent 直取：复用缓存 latent；自动：有缓存时优先 latent。"}),
            "context_sampled_start_tokens": ("INT", {"default": 0, "min": 0, "max": 11, "step": 1,
                "tooltip": "仅用于缓存 latent 直取。让上下文首部的若干 latent token 参与本段采样；0=首部全部固定 Guide。"}),
            "context_sampled_start_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "首部可采样 token 的重绘强度。0=固定不重绘，1=完全重绘；建议 0.20-0.35。"}),
            "context_sampled_tokens": ("INT", {"default": 2, "min": 0, "max": 11, "step": 1,
                "tooltip": "仅用于缓存 latent 直取。让上下文末端的若干 latent token 参与本段采样；0=末端全部固定 Guide。22 帧上下文对应 7 token，默认 2 个（约 5 帧）。"}),
            "context_sampled_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "末端可采样 token 的重绘强度。0=固定不重绘，1=完全重绘；建议 0.20-0.35，过高会造成模糊或跳变。"}),
        }}

    RETURN_TYPES = ("CONDITIONING", "INT", "LATENT")
    RETURN_NAMES = ("条件", "裁剪帧数", "采样潜变量")
    FUNCTION = "apply"
    CATEGORY = "H3 自动导演"

    @staticmethod
    def _attach_stage2_context(conditioning, context_latent):
        if not isinstance(context_latent, dict):
            return conditioning
        stage2 = context_latent.get("h3_stage2_context_latent")
        if stage2 is None:
            return conditioning
        return node_helpers.conditioning_set_values(
            conditioning, {"h3_stage2_context_latent": stage2}
        )

    @staticmethod
    def _direct_latent_context(conditioning, latent, context_latent, context_length, use_audio_context,
                               use_video_context=True, sampled_context_start_tokens=0,
                               sampled_context_start_strength=0.25, sampled_context_tokens=0,
                               sampled_context_strength=0.25):
        """Build native or legacy-compatible context from cached AV latents."""
        target_parts = _av_latent_parts(latent)
        context_parts = _av_latent_parts(context_latent)
        if target_parts is None or context_parts is None:
            raise ValueError("上下文 AV latent 无效，无法直接读取视频 latent")
        target_video, context_video = target_parts[0], context_parts[0]
        if not torch.is_tensor(target_video) or not torch.is_tensor(context_video) or target_video.ndim != 5 or context_video.ndim != 5:
            raise ValueError("H3 视频 latent 必须是 [B,C,T,H,W]")
        if (target_video.shape[0] != context_video.shape[0]
                or target_video.shape[1] != context_video.shape[1]):
            raise ValueError("上下文 latent batch 或通道与当前片段不一致")
        if target_video.shape[3:] != context_video.shape[3:]:
            old_size = tuple(int(value) for value in context_video.shape[3:])
            context_video = _resize_h3_context_latent(
                context_video, int(target_video.shape[3]), int(target_video.shape[4])
            )
            LOG.info(
                "H3 Auto Director: 已将上一段最终二采视频上下文适配到当前一采尺寸：%s -> %s",
                old_size, tuple(int(value) for value in context_video.shape[3:]),
            )
        native_guides = _native_h3_add_guide_supported()
        run = _h3_context_run(context_length)
        steps = {1: 1, 5: 2, 22: 7, 39: 12}[run]
        legacy = None
        if not native_guides:
            from . import legacy_h3_motion
            if not legacy_h3_motion.ensure_legacy_h3_motion_context():
                raise RuntimeError("当前旧版 ComfyUI 无法启用内置 H3 Motion Context 兼容层")
            legacy = legacy_h3_motion
        if not use_video_context:
            if not use_audio_context:
                return conditioning, 0, latent
            audio_tail, audio_steps, _overhang = _h3_audio_tail_from_latent(context_latent, run)
            audio_tail = audio_tail.to(device=target_video.device, dtype=target_video.dtype)
            if native_guides:
                values = {"minimax_keyframes": [{"resolved_frame_index": 0, "audio_latent": audio_tail,
                                                   _NATIVE_CONTEXT_KEY: True}],
                          _NATIVE_CONTEXT_KEY: True,
                          "h3_auto_director_context_run": int(run)}
            else:
                values = {"minimax_refs": [{"kind": "audio", "ref_audio_t": audio_steps,
                                              "audio_latent": audio_tail, legacy.MC_AUDIO_KEY: 0.0}]}
            return _mark_motion_context(node_helpers.conditioning_set_values(conditioning, values)), 0, latent
        if context_video.shape[2] < steps:
            raise ValueError("上下文视频 latent 长度不足，无法读取 %d 帧上下文" % run)
        if run >= _h3_pixel_frames(int(target_video.shape[2])):
            raise ValueError("上下文长度不能占满当前生成片段")
        tail = context_video[:, :, -steps:].to(device=target_video.device, dtype=target_video.dtype)
        # Keep the leading context tokens as immutable Guide rows, while
        # optionally allowing only the tail tokens to be denoised on the
        # current target timeline.  H3's 22-frame context is 7 temporal
        # tokens; the tail is deliberately token-aligned because splitting a
        # token into pixel frames would make PackedLayout disagree with the
        # VAE temporal grid.
        end_tokens = max(0, min(int(sampled_context_tokens or 0), steps))
        start_tokens = max(0, min(int(sampled_context_start_tokens or 0), steps))
        start_strength = max(0.0, min(1.0, float(sampled_context_start_strength)))
        end_strength = max(0.0, min(1.0, float(sampled_context_strength)))
        if start_tokens + end_tokens > steps:
            # Prefer the explicitly requested首部范围 and trim the末端 range
            # when both settings overlap, keeping the mask deterministic.
            end_tokens = max(0, steps - start_tokens)
        sampled_latent = latent
        sampled_token_ranges = []
        if start_tokens:
            sampled_token_ranges.append((0, start_tokens, start_strength))
        if end_tokens:
            sampled_token_ranges.append((steps - end_tokens, steps, end_strength))
        if sampled_token_ranges:
            sampled_video = target_video.clone()
            for range_start, range_end, _strength in sampled_token_ranges:
                sampled_video[:, :, range_start:range_end] = tail[:, :, range_start:range_end].to(
                    device=target_video.device, dtype=target_video.dtype)
            # ComfyUI's denoise_mask uses 0 for an immutable latent and 1 for
            # a fully sampled latent.  The copied context occupies the first
            # ``steps`` temporal tokens of the target, so initialise that
            # window as fixed and leave the rest of the current segment at 1.
            # Previously the whole mask was initialised to 1, which made the
            # copied context outside the selected head/tail ranges fully
            # resample whenever either control was enabled (the controls
            # appeared to work in metadata but had the opposite effect in
            # the sampler).
            video_mask = torch.ones(
                (sampled_video.shape[0], 1, sampled_video.shape[2],
                 sampled_video.shape[3], sampled_video.shape[4]),
                dtype=sampled_video.dtype, device=sampled_video.device,
            )
            video_mask[:, :, :steps] = 0.0
            for range_start, range_end, strength in sampled_token_ranges:
                video_mask[:, :, range_start:range_end] = strength
            source_audio = target_parts[1]
            audio_mask = torch.ones(
                (source_audio.shape[0], 1, source_audio.shape[2], source_audio.shape[3]),
                dtype=source_audio.dtype, device=source_audio.device,
            )
            sampled_latent = dict(latent)
            sampled_latent["samples"] = _h3_av_container(sampled_video, source_audio)
            sampled_latent["noise_mask"] = _h3_av_container(video_mask, audio_mask)
            fixed_tokens = max(0, steps - int(sum(
                end - start for start, end, _ in sampled_token_ranges
            )))
            sampled_latent["h3_auto_director_sampled_context"] = {
                "frames": int(run), "latent_steps": int(sum(end - start for start, end, _ in sampled_token_ranges)),
                "start_tokens": int(start_tokens), "start_strength": start_strength,
                "end_tokens": int(end_tokens), "end_strength": end_strength,
                "fixed_tokens": int(fixed_tokens),
                "video_mask_shape": [int(value) for value in video_mask.shape],
                "video_mask_min": float(video_mask.amin().item()),
                "video_mask_max": float(video_mask.amax().item()),
                "mode": "latent_head_tail",
            }
            LOG.info(
                "H3 Auto Director: latent 上下文首部 %d token(强度=%.3f)、末端 %d token(强度=%.3f)参与采样；"
                "重叠时优先首部范围；固定上下文=%d token，视频 mask 范围=[%.3f, %.3f]",
                int(start_tokens), start_strength, int(end_tokens), end_strength,
                int(fixed_tokens), float(video_mask.amin().item()), float(video_mask.amax().item()),
            )
        if native_guides:
            # Native MiniMaxH3AddGuide represents a guide clip as one
            # keyframe containing its full temporal latent.  Splitting this
            # into arbitrary single-step keyframes can make the prebuilt
            # PackedLayout disagree with cond_video_latents when conditioning
            # is cached or references are also present.
            # AddGuide receives only the fixed middle portion when head/tail
            # tokens are marked sampleable.  Token boundaries are converted
            # back to pixel-frame offsets using H3's temporal packing table.
            try:
                frame_per_token = importlib.import_module("comfy.ldm.minimax.model").FRAME_PER_TOKEN
            except (ImportError, AttributeError):
                frame_per_token = (1, 4, 4, 4, 4)
            token_offsets, cursor = [], 0
            for token_index in range(steps):
                token_offsets.append(cursor)
                cursor += int(frame_per_token[token_index % len(frame_per_token)])
            fixed_start, fixed_end = start_tokens, steps - end_tokens
            keyframes = ([{"resolved_frame_index": int(token_offsets[fixed_start]),
                           "latent": tail[:, :, fixed_start:fixed_end],
                           _NATIVE_CONTEXT_KEY: True}]
                         if fixed_end > fixed_start else [])
            # Do not add a one-token anchor at ``run``.  A temporal token
            # spans multiple pixel frames (the final token covers four), so
            # such an anchor extends the Guide beyond the exact context
            # window and overlaps the first generated frames.  That overlap
            # was the source of a small colour/contrast jump at the join.
            values = {"minimax_keyframes": keyframes, _NATIVE_CONTEXT_KEY: True,
                      "h3_auto_director_context_run": int(run)}
        else:
            offsets, offset = [], 0
            try:
                frame_per_token = importlib.import_module("comfy.ldm.minimax.model").FRAME_PER_TOKEN
            except (ImportError, AttributeError):
                frame_per_token = (1, 4, 4, 4, 4)
            for index in range(steps):
                offsets.append(offset)
                offset += frame_per_token[index % len(frame_per_token)]
            fixed_start, fixed_end = start_tokens, steps - end_tokens
            keyframes = [{"resolved_frame_index": int(frame), legacy.MC_KEY: frame,
                          "latent": tail[:, :, index:index + 1]}
                         for index, frame in enumerate(offsets)
                         if fixed_start <= index < fixed_end]
            values = {"minimax_keyframes": keyframes,
                      "minimax_frame_count": _h3_pixel_frames(int(target_video.shape[2])),
                      "h3_auto_director_context_run": int(run)}
        if use_audio_context:
            audio_tail, audio_steps, overhang = _h3_audio_tail_from_latent(context_latent, run)
            audio_tail = audio_tail.to(device=target_video.device, dtype=target_video.dtype)
            LOG.info(
                "H3 Auto Director: 音频上下文尾部=%d latent 步（来源总长=%d，H3 40Hz，context=%d帧）",
                int(audio_steps), int(context_parts[1].shape[-1]), int(run),
            )
            if native_guides:
                # MiniMaxH3AddGuide anchors audio at the keyframe's frame
                # index. Context audio is the *start* of the carried window,
                # therefore it must start at frame 0 alongside the first
                # video block, not at the final compressed-video block.
                if not keyframes:
                    keyframes.append({"resolved_frame_index": 0, _NATIVE_CONTEXT_KEY: True})
                keyframes[0]["audio_latent"] = audio_tail
            else:
                values["minimax_refs"] = [{"kind": "audio", "ref_audio_t": audio_steps,
                                             "audio_latent": audio_tail,
                                             legacy.MC_AUDIO_KEY: float(run) + float(overhang) / (5.0 / 3.0)}]
        # The guide context is hidden from the delivered clip; SaveSegment
        # removes the same number of decoded frames and audio samples before
        # writing it.
        return _mark_motion_context(node_helpers.conditioning_set_values(conditioning, values)), run, sampled_latent

    @staticmethod
    def _vae_context(conditioning, vae, latent, context_frames, context_length, use_audio_context,
                     context_latent=None):
        """Encode the disk-video tail with the workflow-compatible Guide path.

        This also serves as the compatibility fallback when no cached AV latent
        is available. The native core receives guide frames directly; a legacy
        core receives the internally patched equivalent.
        """
        target_parts = _av_latent_parts(latent)
        if target_parts is None or not torch.is_tensor(target_parts[0]) or target_parts[0].ndim != 5:
            raise ValueError("H3 目标 latent 无效，无法编码视频上下文")
        target_video = target_parts[0]
        run = _h3_context_run(context_length)
        frames = context_frames
        if not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[0] < 1:
            raise ValueError("上下文画面无效，无法回退至 VAE 编码")
        frames = frames[-min(run, int(frames.shape[0])):]
        if frames.shape[0] < run:
            frames = torch.cat((frames[:1].repeat((run - frames.shape[0], 1, 1, 1)), frames), dim=0)
        height, width = int(target_video.shape[3]) * 16, int(target_video.shape[4]) * 16
        samples = frames[..., :3].movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "center").movedim(1, -1)
        encoded = _encode_h3_video(vae, samples)
        if not torch.is_tensor(encoded) or encoded.ndim != 5:
            raise ValueError("H3 视频 VAE 未返回 [B,C,T,H,W] latent")
        steps = min(int(encoded.shape[2]), int(target_video.shape[2]))
        sampled_latent = latent
        if _native_h3_add_guide_supported():
            keyframes = [{"resolved_frame_index": 0, "latent": encoded[:, :, :steps].to(
                device=target_video.device, dtype=target_video.dtype), _NATIVE_CONTEXT_KEY: True}]
            values = {"minimax_keyframes": keyframes, _NATIVE_CONTEXT_KEY: True,
                      "h3_auto_director_context_run": int(run)}
            if bool(use_audio_context) and context_latent is not None:
                try:
                    audio_tail, _audio_steps, _overhang = _h3_audio_tail_from_latent(context_latent, run)
                    keyframes[0]["audio_latent"] = audio_tail.to(
                        device=target_video.device, dtype=target_video.dtype)
                except (ValueError, RuntimeError) as exc:
                    LOG.warning("H3 Auto Director: 帧 Guide 音频上下文不可用，保留画面上下文：%s", exc)
            LOG.info("H3 Auto Director: 使用新版原生 Guide VAE 回退上下文：视频 keyframe=%d，音频上下文不可用", len(keyframes))
            return _mark_motion_context(node_helpers.conditioning_set_values(conditioning, values)), run, sampled_latent
        from . import legacy_h3_motion
        if not legacy_h3_motion.ensure_legacy_h3_motion_context():
            raise RuntimeError("当前旧版 ComfyUI 无法启用内置 H3 Motion Context 兼容层")
        offsets, current = [], 0
        module = importlib.import_module("comfy.ldm.minimax.model")
        for index in range(steps):
            offsets.append(current)
            current += module.FRAME_PER_TOKEN[index % len(module.FRAME_PER_TOKEN)]
        keyframes = [{"resolved_frame_index": 0, legacy_h3_motion.MC_KEY: frame,
                      "latent": encoded[:, :, index:index + 1].to(device=target_video.device, dtype=target_video.dtype)}
                     for index, frame in enumerate(offsets)]
        values = {"minimax_keyframes": keyframes,
                  "minimax_frame_count": _h3_pixel_frames(int(target_video.shape[2])),
                  "h3_auto_director_context_run": int(run)}
        if bool(use_audio_context) and context_latent is not None:
            try:
                audio_tail, audio_steps, overhang = _h3_audio_tail_from_latent(context_latent, run)
                values["minimax_refs"] = [{"kind": "audio", "ref_audio_t": audio_steps,
                                             "audio_latent": audio_tail,
                                             legacy_h3_motion.MC_AUDIO_KEY: float(run) + float(overhang) / (5.0 / 3.0)}]
            except (ValueError, RuntimeError) as exc:
                LOG.warning("H3 Auto Director: 旧版帧 Guide 音频上下文不可用，保留画面上下文：%s", exc)
        LOG.info("H3 Auto Director: 使用内置旧版 Motion Context VAE 回退：视频 keyframe=%d，音频上下文不可用", len(keyframes))
        return _mark_motion_context(node_helpers.conditioning_set_values(conditioning, values)), run, sampled_latent

    def apply(self, conditioning, vae, latent, context_frames, use_video_context, use_audio_context,
              context_length, context_latent=None, use_video_latent=True,
              context_method="工作流视频帧 Guide", context_sampled_start_tokens=0,
              context_sampled_start_strength=0.25, context_sampled_tokens=2,
              context_sampled_strength=0.25, **_legacy_noise):
        global _LAST_MOTION_CONTEXT_TRIM
        stage2_context = context_latent.get("h3_stage2_context_latent") if isinstance(context_latent, dict) else None
        method = str(context_method or "工作流视频帧 Guide")
        def with_stage2(value):
            global _LAST_MOTION_CONTEXT_TRIM
            context_trim = int(value[1]) if isinstance(value, (list, tuple)) and len(value) > 1 else 0
            sampling_latent = value[2] if isinstance(value, (list, tuple)) and len(value) > 2 else latent
            _LAST_MOTION_CONTEXT_TRIM = max(0, context_trim)
            if context_trim > 0:
                LOG.info("H3 Auto Director: 上下文裁剪=%d 帧",
                         _LAST_MOTION_CONTEXT_TRIM)
            conditioned = value[0] if isinstance(value, (list, tuple)) and value else value
            if stage2_context is not None:
                conditioned = self._attach_stage2_context(conditioned, stage2_context)
            return (conditioned, _LAST_MOTION_CONTEXT_TRIM, sampling_latent)
        if not use_video_context:
            if bool(use_audio_context) and context_latent is not None:
                try:
                    result = self._direct_latent_context(conditioning, latent, context_latent, context_length,
                                                         True, use_video_context=False,
                                                         sampled_context_start_tokens=0,
                                                         sampled_context_start_strength=context_sampled_start_strength,
                                                         sampled_context_tokens=0,
                                                         sampled_context_strength=context_sampled_strength)
                    LOG.info("H3 Auto Director: 应用音频上下文（无视频 keyframe），音频 ref=%d，ref_audio_t=%d",
                             sum(1 for entry in result[0] if isinstance(entry, (list, tuple))
                                 for value in [entry[1]] if isinstance(value, dict)
                                 for ref in (value.get("minimax_refs") or [])
                                 if isinstance(ref, dict) and ref.get("audio_latent") is not None),
                             int(result[1]))
                    return with_stage2(result)
                except (ValueError, RuntimeError) as exc:
                    LOG.info("H3 Auto Director: 音频上下文直取不可用，跳过音频上下文：%s", exc)
            return with_stage2((self._attach_stage2_context(conditioning, stage2_context), 0))
        deferred_decode = bool(isinstance(context_latent, dict) and context_latent.get("h3_deferred_decode"))
        use_direct_latent = ((bool(use_video_latent) or deferred_decode) and context_latent is not None
                             and (method == "缓存视频 latent 直取"
                                  or method == "自动（latent 优先）"
                                  or deferred_decode))
        if deferred_decode and method != "缓存视频 latent 直取":
            LOG.info("H3 Auto Director: 已开启所有片段采样完成后统一解码，视频上下文强制使用缓存潜空间直取")
        # A stage-one cache may predate the separate context-video directory.
        # If the workflow-compatible frame input is only the inert placeholder,
        # transparently fall back to its cached latent instead of encoding a
        # 1x1 black frame as a false continuation source.
        if (not use_direct_latent and context_latent is not None
                and torch.is_tensor(context_frames) and context_frames.ndim == 4
                and (int(context_frames.shape[0]) <= 1 or int(context_frames.shape[1]) <= 1
                     or int(context_frames.shape[2]) <= 1)):
            use_direct_latent = bool(use_video_latent)
            if use_direct_latent:
                LOG.warning("H3 Auto Director: 上下文视频缓存不可用，回退到 AV latent 直取")
        if (int(context_sampled_start_tokens or 0) > 0 or int(context_sampled_tokens or 0) > 0) and not use_direct_latent:
            LOG.info(
                "H3 Auto Director: 已设置首部/末端可采样 token=%d/%d，但当前为视频帧 Guide 路径；"
                "该参数仅在‘缓存视频 latent 直取’或‘自动（latent 优先）’路径生效",
                int(context_sampled_start_tokens), int(context_sampled_tokens),
            )
        if use_direct_latent:
            try:
                result = self._direct_latent_context(
                    conditioning, latent, context_latent, context_length, use_audio_context,
                    sampled_context_start_tokens=context_sampled_start_tokens,
                    sampled_context_start_strength=context_sampled_start_strength,
                    sampled_context_tokens=context_sampled_tokens,
                    sampled_context_strength=context_sampled_strength,
                )
                keyframe_count = sum(1 for entry in result[0] if isinstance(entry, (list, tuple))
                                     for value in [entry[1]] if isinstance(value, dict)
                                     for kf in (value.get("minimax_keyframes") or []) if isinstance(kf, dict))
                audio_count = sum(1 for entry in result[0] if isinstance(entry, (list, tuple))
                                  for value in [entry[1]] if isinstance(value, dict)
                                  for ref in (value.get("minimax_refs") or [])
                                  if isinstance(ref, dict) and ref.get("audio_latent") is not None)
                LOG.info("H3 Auto Director: 应用 latent 直取上下文：视频 keyframe=%d，音频 ref=%d，context_length=%d",
                         keyframe_count, audio_count, int(result[1]))
                return with_stage2(result)
            except ValueError as exc:
                LOG.info("H3 Auto Director: 视频 latent 直取不可用，回退 VAE 上下文编码：%s", exc)
        result = self._vae_context(conditioning, vae, latent, context_frames, context_length,
                                    use_audio_context, context_latent=context_latent)
        return with_stage2(result)


def _normalize_h3_audio_latent(latent, where="音频 latent"):
    """Validate and normalize the H3 audio stream before decode/save/reuse."""
    if not torch.is_tensor(latent):
        raise ValueError("%s 必须是 Tensor，实际类型：%s" % (where, type(latent).__name__))
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)
    if latent.ndim != 4:
        raise ValueError("%s 必须是 [B,32,2,T]，实际形状：%s" % (where, tuple(latent.shape)))
    if int(latent.shape[1]) != 32 or int(latent.shape[2]) != 2:
        raise ValueError("%s 必须是 [B,32,2,T]，实际形状：%s；请确认连接的是 H3 音频 VAE/联合 latent" % (where, tuple(latent.shape)))
    if not torch.isfinite(latent).all():
        raise ValueError("%s 包含 NaN/Inf，无法用于 H3 音频解码" % where)
    return latent


def _log_h3_audio_vae(audio_vae, name=""):
    """Log the VAE contract so a wrong audio VAE is visible before sampling."""
    rate = getattr(audio_vae, "audio_sample_rate_output",
                   getattr(audio_vae, "audio_sample_rate", None))
    channels = getattr(audio_vae, "latent_channels", None)
    LOG.info("H3 Auto Director: 音频 VAE=%s sample_rate=%s latent_channels=%s",
             str(name), rate, channels)
    if rate not in (None, 32000):
        LOG.warning("H3 Auto Director: 当前音频 VAE 采样率为 %s（H3 应为 32000 Hz），可能导致音频失真或杂音", rate)
    if channels not in (None, 32):
        LOG.warning("H3 Auto Director: 当前音频 VAE latent_channels=%s（H3 应为 32），请确认使用 minimax_h3_audio_vae_fp32", channels)


def _decode_h3_audio(audio_vae, latent):
    """Return ComfyUI AUDIO waveform as [B, C, L] at the H3 VAE rate.

    ``VAE.decode`` historically returns audio as [B, L, C] because it uses
    the same last-channel convention as image/video output.  A few older
    audio VAE wrappers return [B, C, L] directly, so detect both layouts
    instead of blindly transposing one of them into a noise-like stream.
    """
    latent = _normalize_h3_audio_latent(latent, "H3 音频 latent")
    decoded = audio_vae.decode(latent)
    if not torch.is_tensor(decoded) or decoded.ndim != 3:
        raise ValueError("H3 音频 VAE 解码结果必须是三维 waveform，实际形状：%s" % (tuple(decoded.shape) if torch.is_tensor(decoded) else type(decoded).__name__))
    if decoded.shape[1] <= 8 and decoded.shape[2] > decoded.shape[1]:
        waveform = decoded
    elif decoded.shape[2] <= 8:
        waveform = decoded.movedim(-1, 1)
    else:
        raise ValueError("无法判断 H3 音频 waveform 的声道维度，实际形状：%s" % (tuple(decoded.shape),))
    waveform = waveform.detach().to(dtype=torch.float32).contiguous()
    sample_rate = int(getattr(audio_vae, "audio_sample_rate_output",
                              getattr(audio_vae, "audio_sample_rate", 32000)) or 32000)
    if sample_rate <= 0:
        sample_rate = 32000
    if waveform.shape[1] > 2:
        LOG.warning("H3 Auto Director: 音频解码得到 %d 个声道，截取前 2 个声道", waveform.shape[1])
        waveform = waveform[:, :2]
    LOG.debug("H3 Auto Director: 音频解码 shape=%s sample_rate=%d latent=%s min=%.5f max=%.5f rms=%.5f",
              tuple(waveform.shape), sample_rate, tuple(latent.shape),
              float(waveform.min().detach().cpu()), float(waveform.max().detach().cpu()),
              float(waveform.square().mean().sqrt().detach().cpu()))
    return waveform, sample_rate


def _write_wav(path: Path, audio):
    waveform = audio.get("waveform")
    if waveform is None:
        return None
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        raise ValueError("保存音频需要 [B,C,L] waveform")
    # Normalize legacy custom-node AUDIO values before PCM interleaving.
    if waveform.shape[1] > 8 and waveform.shape[2] <= 8:
        waveform = waveform.movedim(-1, 1)
    if waveform.shape[1] < 1 or waveform.shape[1] > 8:
        raise ValueError("音频 waveform 声道维度无效：%s" % (tuple(waveform.shape),))
    samples = waveform[0].detach().cpu().to(dtype=torch.float32).clamp(-1, 1).movedim(0, 1).numpy()
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    sample_rate = int(audio.get("sample_rate", 32000) or 32000)
    if sample_rate <= 0:
        sample_rate = 32000
    with wave.open(str(path), "wb") as out:
        out.setnchannels(samples.shape[1])
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    LOG.debug("H3 Auto Director: WAV 写入 %s channels=%d samples=%d sample_rate=%d",
              path, samples.shape[1], samples.shape[0], sample_rate)
    return path


def _ffmpeg_encoders(ffmpeg):
    """Return the encoder names advertised by this ffmpeg build."""
    try:
        result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return set()
    if result.returncode != 0:
        return set()
    names = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) >= 2 and not parts[0].startswith("--"):
            names.add(parts[1])
    return names


def _quality_args(codec, device, quality):
    quality = quality if quality in QUALITY_CHOICES else QUALITY_CHOICES[0]
    rank = QUALITY_CHOICES.index(quality)
    if device == "gpu":
        cq = ("16", "19", "23", "28")[rank]
        if codec in {"h264", "hevc", "av1"}:
            return ["-cq", cq, "-preset", ("p7", "p5", "p3", "p1")[rank]]
        return []
    if codec == "h264":
        return ["-crf", ("16", "18", "22", "28")[rank], "-preset", ("veryslow", "slower", "medium", "veryfast")[rank]]
    if codec == "hevc":
        return ["-crf", ("18", "20", "24", "30")[rank], "-preset", ("slower", "medium", "medium", "fast")[rank]]
    if codec == "vp9":
        return ["-crf", ("18", "22", "28", "36")[rank], "-b:v", "0", "-deadline", "good", "-cpu-used", str(min(rank, 3))]
    return ["-crf", ("18", "22", "28", "36")[rank], "-cpu-used", str(min(rank, 3))]


def _run_ffmpeg_raw(ffmpeg, output, arr, fps, video_format, codec, encoder, device, quality):
    if arr.ndim != 4 or arr.shape[0] < 1 or arr.shape[-1] < 3:
        raise ValueError("保存视频需要 [帧,高,宽,RGB] 图像序列，实际形状：%s" % (tuple(arr.shape),))
    h, w = int(arr.shape[1]), int(arr.shape[2])
    muxer = VIDEO_FORMATS.get(video_format, "mp4")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(float(fps)), "-i", "pipe:0", "-an", "-c:v", encoder]
    command.extend(_quality_args(codec, device, quality))
    command.extend(["-pix_fmt", "yuv420p"])
    if muxer in {"mp4", "mov"}:
        command.extend(["-movflags", "+faststart"])
    command.extend(["-f", muxer, str(output)])
    try:
        result = subprocess.run(command, input=arr.tobytes(), capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        output.unlink(missing_ok=True)
        raise RuntimeError(str(exc)) from exc
    if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        detail = (result.stderr or b"").decode(errors="replace")[-1200:]
        output.unlink(missing_ok=True)
        raise RuntimeError(detail or f"ffmpeg encoder {encoder} failed")


def _verify_video_stream(path: Path):
    """Reject audio-only/corrupt outputs before they become a context source."""
    if av is None:
        return
    try:
        with av.open(str(path), "r") as container:
            streams = tuple(container.streams.video)
            if not streams:
                raise ValueError("文件没有视频流")
            frames = 0
            for _ in container.decode(streams[0]):
                frames += 1
                if frames:
                    break
            if frames < 1:
                raise ValueError("视频流没有可解码帧")
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError("保存的视频无效，不能作为接续上下文：%s（%s）" % (path, exc)) from exc


def _encode_video_with_fallback(path: Path, images, fps, video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量"):
    """Encode once per candidate, with a finite GPU-to-CPU fallback."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法保存视频。请重启 ComfyUI，或设置环境变量 FFMPEG_PATH 指向 ffmpeg.exe。")
    fmt = str(video_format or "mp4").lower().lstrip(".")
    codec = str(video_codec or "h264").lower()
    if fmt not in VIDEO_FORMATS:
        fmt = "mp4"
    if codec not in VIDEO_CODECS:
        codec = "h264"
    device = "gpu" if str(encoder_device or "CPU").upper() == "GPU" else "cpu"
    arr = images.detach().cpu().clamp(0, 1).mul(255).byte().numpy()[..., :3]
    advertised = _ffmpeg_encoders(ffmpeg)
    candidates = []
    if device == "gpu":
        candidates.extend(name for name in VIDEO_CODECS[codec]["gpu"] if not advertised or name in advertised)
    candidates.append(VIDEO_CODECS[codec]["cpu"])
    # De-duplicate while preserving order. This is deliberately iterative: a
    # failed GPU encoder can never schedule itself again.
    unique = list(dict.fromkeys(candidates))
    errors = []
    for encoder in unique:
        path.unlink(missing_ok=True)
        try:
            _run_ffmpeg_raw(ffmpeg, path, arr, fps, fmt, codec, encoder, device if encoder != VIDEO_CODECS[codec]["cpu"] else "cpu", quality)
            return encoder
        except RuntimeError as exc:
            errors.append(f"{encoder}: {exc}")
            path.unlink(missing_ok=True)
    raise RuntimeError("视频编码失败（GPU失败后已回退CPU，未继续重试）：\n" + "\n".join(errors)[-3000:])


def _encode_concat_with_fallback(ffmpeg, list_path, output, video_format, video_codec, encoder_device, quality):
    fmt = str(video_format or "mp4").lower().lstrip(".")
    codec = str(video_codec or "h264").lower()
    device = "gpu" if str(encoder_device or "CPU").upper() == "GPU" else "cpu"
    advertised = _ffmpeg_encoders(ffmpeg)
    candidates = []
    if device == "gpu":
        candidates.extend(name for name in VIDEO_CODECS[codec]["gpu"] if not advertised or name in advertised)
    candidates.append(VIDEO_CODECS[codec]["cpu"])
    errors = []
    for encoder in list(dict.fromkeys(candidates)):
        output.unlink(missing_ok=True)
        audio_codec = "libopus" if fmt == "webm" else "aac"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
                   "-map", "0:v:0", "-map", "0:a:0?", "-c:v", encoder]
        command.extend(_quality_args(codec, device if encoder != VIDEO_CODECS[codec]["cpu"] else "cpu", quality))
        command.extend(["-pix_fmt", "yuv420p", "-c:a", audio_codec, "-shortest"])
        if fmt in {"mp4", "mov"}:
            command.extend(["-movflags", "+faststart"])
        command.extend(["-f", VIDEO_FORMATS.get(fmt, "mp4"), str(output)])
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            result = None
            detail = str(exc)
        else:
            detail = (result.stderr or "")[-1600:]
        if result is not None and result.returncode == 0 and output.is_file() and output.stat().st_size:
            return encoder
        output.unlink(missing_ok=True)
        errors.append(f"{encoder}: {detail}")
    raise RuntimeError("最终视频编码失败（GPU失败后已回退CPU，未继续重试）：\n" + "\n".join(errors)[-3500:])


def _trim_disk_video(path: Path, output: Path, trim_frames: int, fps=FPS,
                     video_format="mp4", video_codec="h264", quality="最高质量"):
    """Decode a saved segment, remove its context prefix, and re-encode it.

    This is intentionally used only during final assembly (when all segment
    files are already durable).  The normal per-segment save path remains
    lossless with respect to its selected encoder and keeps the untrimmed
    context source separately.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法在最终拼接前裁剪磁盘片段")
    trim_frames = max(0, int(trim_frames or 0))
    if trim_frames <= 0:
        shutil.copy2(path, output)
        return
    fmt = str(video_format or "mp4").lower().lstrip(".")
    codec = str(video_codec or "h264").lower()
    encoder = VIDEO_CODECS.get(codec, VIDEO_CODECS["h264"])["cpu"]
    start_seconds = float(trim_frames) / float(fps or FPS)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
               "-map", "0:v:0", "-map", "0:a:0?",
               "-vf", f"trim=start_frame={trim_frames},setpts=PTS-STARTPTS",
               "-af", f"atrim=start={start_seconds:.9f},asetpts=PTS-STARTPTS",
               "-c:v", encoder]
    command.extend(_quality_args(codec, "cpu", quality))
    command.extend(["-pix_fmt", "yuv420p", "-c:a", ("libopus" if fmt == "webm" else "aac"),
                    "-shortest"])
    if fmt in {"mp4", "mov"}:
        command.extend(["-movflags", "+faststart"])
    command.extend(["-f", VIDEO_FORMATS.get(fmt, "mp4"), str(output)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("最终拼接前裁剪片段失败：%s" % ((result.stderr or "")[-1600:]))
    _verify_video_stream(output)


def _write_segment_video(path: Path, images, audio, fps, video_format, video_codec, encoder_device, quality):
    """Encode an image sequence and mux its optional audio atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = path.with_name(path.stem + ".video_tmp" + path.suffix)
    _encode_video_with_fallback(tmp_video, images, fps, video_format, video_codec, encoder_device, quality)
    tmp_wav = path.with_suffix(".audio_tmp.wav")
    wav = _write_wav(tmp_wav, audio) if audio is not None else None
    if wav is None:
        os.replace(tmp_video, path)
        _verify_video_stream(path)
        return
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        tmp_video.unlink(missing_ok=True)
        tmp_wav.unlink(missing_ok=True)
        raise RuntimeError("未找到 ffmpeg，无法封装 H3 片段音频。请重启 ComfyUI，或设置环境变量 FFMPEG_PATH 指向 ffmpeg.exe。")
    try:
        audio_codec = "libopus" if str(video_format).lower() == "webm" else "aac"
        mux_args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp_video), "-i", str(wav),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", audio_codec, "-shortest"]
        if str(video_format).lower() in {"mp4", "mov"}:
            mux_args.extend(["-movflags", "+faststart"])
        mux_args.append(str(path))
        subprocess.run(mux_args, check=True, capture_output=True)
        _verify_video_stream(path)
    except subprocess.CalledProcessError as exc:
        path.unlink(missing_ok=True)
        detail = (exc.stderr or b"").decode(errors="replace")[-1200:]
        raise RuntimeError("ffmpeg 无法封装片段音频：%s" % detail) from exc
    finally:
        tmp_video.unlink(missing_ok=True)
        tmp_wav.unlink(missing_ok=True)


def _trim_context_prefix(images, audio, trim_frames, fps=FPS):
    """Remove the pinned previous-clip head from both output streams.

    H3 Guide writes context into the beginning of the requested timeline. The
    first frame is intentionally shared with the previous clip, so saving the
    full decoded result would duplicate the context window at every join.
    """
    count = max(0, int(trim_frames or 0))
    if count <= 0:
        return images, audio
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("上下文裁剪需要 [T,H,W,C] 视频画面")
    if count >= int(images.shape[0]):
        raise ValueError("上下文裁剪帧数 %d 不得达到视频总帧数 %d" % (count, int(images.shape[0])))
    trimmed_images = images[count:]
    trimmed_audio = audio
    if isinstance(audio, dict) and torch.is_tensor(audio.get("waveform")):
        waveform = audio["waveform"]
        sample_rate = int(audio.get("sample_rate", 32000) or 32000)
        cut = int(round(count / float(fps) * sample_rate))
        if waveform.ndim != 3:
            raise ValueError("上下文裁剪需要 [B,C,L] 音频 waveform")
        if cut >= int(waveform.shape[-1]):
            raise ValueError("上下文裁剪音频后没有剩余样本")
        waveform = waveform[..., cut:]
        wanted = int(round((int(trimmed_images.shape[0]) / float(fps)) * sample_rate))
        if waveform.shape[-1] > wanted:
            waveform = waveform[..., :wanted]
        trimmed_audio = dict(audio)
        trimmed_audio["waveform"] = waveform
    return trimmed_images, trimmed_audio


def _default_context_trim_frames(plan, segment_index):
    """Infer the hidden guide-context prefix for workflows without a trim socket."""
    index = int(segment_index)
    if index <= 1 or not _video_context_enabled(plan):
        return 0
    segment = _segment(plan, index)
    if _use_previous_video_reference(plan, index) or not bool(segment.get("continue_video", True)):
        return 0
    configured = max(0, int(plan.get("auto_context_crop_frames", 0) or 0))
    if configured > 0:
        return configured
    run = _h3_context_run(plan.get("_runtime_context_length", FRAME_CONTEXT_DEFAULT))
    return run


def _auto_context_crop_enabled():
    value = os.environ.get("H3_AUTO_DIRECTOR_AUTO_CROP")
    if value is None:
        return AUTO_CONTEXT_CROP_DEFAULT
    return str(value).strip().lower() in {"1", "true", "yes", "on", "开启", "是"}


class H3AutoDirectorSaveSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("H3_AUTO_PLAN",), "segment_index": ("INT", {"default": 1, "min": 1}),
            "latent": ("LATENT",), "images": ("IMAGE",), "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            "output_root": ("STRING", {"default": "", "tooltip": "中间片段文件名，不含扩展名；留空使用 H3"}),
            "video_format": (list(VIDEO_FORMATS), {"default": "mp4"}),
            "video_codec": (list(VIDEO_CODECS), {"default": "h264"}),
            "encoder_device": (list(ENCODER_DEVICES), {"default": "CPU"}),
            "quality": (list(QUALITY_CHOICES), {"default": "最高质量"}),
            "color_correction": (list(COLOR_CORRECTION_CHOICES), {"default": "关闭"}),
        }, "optional": {
            "trim_frames": ("INT", {"default": 0, "min": 0, "max": 4096,
                              "tooltip": "去除上一段上下文固定在本段开头的帧，避免拼接重复；连接运动上下文节点的裁剪帧数输出。"}),
            "auto_context_crop": ("BOOLEAN", {"default": False,
                                  "label_on": "开启", "label_off": "关闭",
                                  "tooltip": "开启后保存时自动去除上下文固定在片段开头的帧；关闭则保留完整解码时间轴。"}),
            "audio": ("AUDIO",),
            "stage1_latent": ("LATENT",),
            "stage1_images": ("IMAGE",),
            "scene_cut_protection": ("BOOLEAN", {"default": True, "tooltip": "检测到明显场景切换时跳过颜色匹配，避免把新场景强行拉回旧场景。"}),
            "scene_cut_threshold": ("FLOAT", {"default": 0.18, "min": 0.02, "max": 1.0, "step": 0.01}),
            "correction_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
            "residual_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("已保存视频", "已保存潜变量")
    FUNCTION = "save"
    CATEGORY = "H3 自动导演"
    OUTPUT_NODE = True

    def save(self, plan, segment_index, latent, images, fps, output_root="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量", color_correction="关闭", scene_cut_protection=True, scene_cut_threshold=0.18, correction_strength=0.75, residual_strength=0.2, trim_frames=0, auto_context_crop=False, audio=None, stage1_latent=None, stage1_images=None):
        global _LAST_STAGE1_CONTEXT
        # The dual sampler attaches this only when "最终仅使用一采音频"
        # is enabled.  Prefer it over a separately connected AUDIO decode so
        # both the context copy and the user-facing saved clip carry stage-one
        # audio even in graphs with stale/miswired audio connections.
        stage1_audio_override = latent.get("_h3_stage1_audio") if isinstance(latent, dict) else None
        if isinstance(stage1_audio_override, dict) and stage1_audio_override.get("waveform") is not None:
            audio = stage1_audio_override
            LOG.info("H3 Auto Director: 保存第 %d 段时使用一采音频覆盖外部 AUDIO 输入", int(segment_index))
        global _LAST_MOTION_CONTEXT_TRIM
        deferred_decode = bool(plan.get("decode_after_all_segments", False))
        requested_trim = int(trim_frames or 0)
        motion_trim = _LAST_MOTION_CONTEXT_TRIM
        configured_trim = max(0, int(plan.get("auto_context_crop_frames", 0) or 0))
        automatic_trim = _default_context_trim_frames(plan, segment_index)
        if requested_trim <= 0 and configured_trim > 0 and automatic_trim > 0:
            # A project-level value overrides the Motion Context node's
            # calculated default while preserving explicit no-context zeros.
            requested_trim = configured_trim
        elif requested_trim <= 0 and motion_trim is not None:
            # A connected Motion Context output of zero is intentional for
            # audio-only/no-context segments. Do not infer a video context
            # from the project defaults in that case.
            requested_trim = max(0, int(motion_trim))
        if requested_trim <= 0 and motion_trim is None:
            requested_trim = automatic_trim
        # The node switch is authoritative.  The environment variable remains
        # available only for older programmatic callers that pass None.
        crop_enabled = (_auto_context_crop_enabled() if auto_context_crop is None
                        else bool(auto_context_crop)) or configured_trim > 0
        if not crop_enabled:
            if requested_trim > 0:
                LOG.info(
                    "H3 Auto Director: 自动裁剪暂时关闭，保留第 %d 段完整上下文时间轴（原计划裁剪 %d 帧）",
                    int(segment_index), int(requested_trim),
                )
            requested_trim = 0
        if not deferred_decode:
            images, audio = _trim_context_prefix(images, audio, requested_trim, FPS)
        if requested_trim > 0:
            LOG.info("H3 Auto Director: 保存第 %d 段前裁剪上下文头部 %d 帧，避免拼接重复", int(segment_index), requested_trim)
        if stage1_latent is None:
            stage1_latent = latent.get("_h3_stage1_context") if isinstance(latent, dict) else None
        if stage1_latent is None:
            # Existing workflows predate the optional socket.  The dual
            # sampler publishes the first-pass result for this immediately
            # following save node, so old graphs still receive the separate
            # stage-one cache automatically.
            stage1_latent = _LAST_STAGE1_CONTEXT
        try:
            requested_fps = float(fps)
        except (TypeError, ValueError):
            requested_fps = FPS
        # MiniMax H3's latent timeline is trained and decoded at 24 fps.
        # Encoding it at an arbitrary container fps changes playback speed;
        # use 24 consistently for muxing and context extraction instead.
        if abs(requested_fps - FPS) > 1e-6:
            LOG.warning("H3 Auto Director: 忽略 %.3f fps 设置，H3 视频固定以 %.0f fps 保存", requested_fps, FPS)
        fps = FPS
        plan = _runtime_plan(plan)
        video_path, latent_path = _paths(plan, int(segment_index), output_root, video_format, for_write=True)
        context_path = Path(plan["project_dir"]) / CONTEXT_DIR_NAME / video_path.name
        video_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.parent.mkdir(parents=True, exist_ok=True)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        if st_save is None:
            raise RuntimeError("safetensors is required to save H3 AV context")
        parts = list(latent["samples"].unbind()) if hasattr(latent["samples"], "unbind") else list(latent["samples"])
        if len(parts) < 2:
            raise ValueError("Sampler output must be an H3 video/audio latent pair")
        parts[1] = _normalize_h3_audio_latent(parts[1], "保存前音频 latent")
        images_to_save = images
        segment_number = int(segment_index)
        correction_mode = str(color_correction)
        if deferred_decode:
            LOG.info("H3 Auto Director: 统一解码模式跳过保存节点校色，最终逐段解码时使用原始 latent")
        elif correction_mode in {"匹配首段", "匹配上段"} and segment_number <= 1:
            LOG.info("H3 Auto Director: color correction skipped for first segment")
        elif correction_mode in {"匹配首段", "匹配上段"}:
            anchor_path = _color_reference_path(plan, segment_number, correction_mode, output_root, video_format)
            if anchor_path is None:
                LOG.warning("H3 Auto Director: color correction skipped; anchor for segment %d is missing", segment_number)
            else:
                try:
                    anchor = _load_context_video(anchor_path, max_frames=39)
                    images_to_save, color_info = _color_match_to_reference(
                        images, anchor, blend=correction_strength,
                        scene_cut_protection=scene_cut_protection,
                        scene_cut_threshold=scene_cut_threshold,
                        residual_strength=residual_strength, return_info=True)
                    if color_info["scene_cut"]:
                        LOG.info("H3 Auto Director: segment %d detected a scene cut; color matching skipped", segment_number)
                    elif color_info["correction_applied"]:
                        LOG.info("H3 Auto Director: matched segment %d colors to %s", segment_number, anchor_path)
                except Exception as exc:
                    LOG.warning("H3 Auto Director: color correction skipped for segment %d: %s", segment_number, exc)
        if deferred_decode:
            # In unified-decode mode, latent is the only per-segment durable
            # source. This prevents repeated VAE encode/decode and keeps GPU
            # memory bounded to one segment at final assembly.
            context_path = Path("")
            video_path = Path("")
            LOG.info("H3 Auto Director: 第 %d 段仅保存最终 AV latent，等待全部片段完成后逐段解码", int(segment_index))
        else:
            # Context is intentionally written from raw decoded frames. The
            # next segment must never inherit display-only color correction.
            _write_segment_video(context_path, images, audio, fps, video_format, video_codec, encoder_device, quality)
            if images_to_save is images:
                shutil.copy2(context_path, video_path)
            else:
                _write_segment_video(video_path, images_to_save, audio, fps, video_format, video_codec, encoder_device, quality)
        st_save({"video": parts[0].detach().cpu().contiguous(), "audio": parts[1].detach().cpu().contiguous()}, str(latent_path), metadata={"format": "h3_auto_director_av_v1", "segment_index": str(int(segment_index))})
        stage1_cache_path = None
        if stage1_latent is not None:
            stage1_parts = _av_latent_parts(stage1_latent)
            if stage1_parts is not None:
                stage1_audio = _normalize_h3_audio_latent(stage1_parts[1], "保存前一采音频 latent")
                _, stage1_cache = _paths(plan, int(segment_index), output_root, video_format,
                                         for_write=True, for_context=True, context_stage=1)
                stage1_cache_path = stage1_cache
                stage1_cache.parent.mkdir(parents=True, exist_ok=True)
                st_save({"video": stage1_parts[0].detach().cpu().contiguous(),
                         "audio": stage1_audio.detach().cpu().contiguous()},
                        str(stage1_cache), metadata={"format": "h3_auto_director_av_stage1_v1", "segment_index": str(int(segment_index))})
                if stage1_images is not None and torch.is_tensor(stage1_images) and stage1_images.numel():
                    stage1_images, _ = _trim_context_prefix(stage1_images, None, requested_trim, FPS)
                    stage1_video, _ = _paths(plan, int(segment_index), output_root, video_format,
                                             for_write=True, for_context=True, context_stage=1)
                    stage1_video.parent.mkdir(parents=True, exist_ok=True)
                    _write_segment_video(stage1_video, stage1_images, audio, fps,
                                         video_format, video_codec, encoder_device, quality)
                LOG.info("H3 Auto Director: 已保存第 %d 段一采上下文 latent 与二采上下文 latent", int(segment_index))
        state_path = _state_path(plan)
        state = _load_json(state_path, {"version": 3, "segments": {}})
        segment_state = state.setdefault("segments", {}).setdefault(str(int(segment_index)), {})
        segment_state.update({
            "status": "completed",
            "video": str(video_path) if not deferred_decode else "",
            "context_video": str(context_path) if not deferred_decode else "",
            "latent": str(latent_path),
            "stage1_latent": str(stage1_cache_path) if stage1_cache_path is not None else "",
            "context_trim_frames": int(requested_trim),
            "fps": float(fps),
        })
        state["last_completed"] = int(segment_index)
        _atomic_json(state_path, state)
        _LAST_STAGE1_CONTEXT = None
        _LAST_MOTION_CONTEXT_TRIM = None
        saved_video = "已缓存最终潜空间（等待统一解码）" if deferred_decode else str(video_path)
        return (saved_video, str(latent_path))


class H3AutoDirectorController:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("H3_AUTO_PLAN",), "segment_index": ("INT", {"default": 1, "min": 1}), "saved_video": ("STRING",),
            "segment_node_id": ("STRING", {"forceInput": True}),
            "output_root": ("STRING", {"default": "", "tooltip": "最终视频文件名，不含扩展名；留空使用 H3"}),
            "video_format": (list(VIDEO_FORMATS), {"default": "mp4"}),
            "video_codec": (list(VIDEO_CODECS), {"default": "h264"}),
            "encoder_device": (list(ENCODER_DEVICES), {"default": "CPU"}),
            "quality": (list(QUALITY_CHOICES), {"default": "最高质量"}),
        }, "optional": {
            "video_vae": ("VAE",), "audio_vae": ("VAE",),
            "cleanup_after_final": ("BOOLEAN", {"default": True, "tooltip": "仅在最终视频拼接成功且文件确认非空后清理显存。"}),
        }, "hidden": {"prompt": "PROMPT", "client_id": "CLIENT_ID"}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("运行状态", "最终视频")
    FUNCTION = "advance"
    CATEGORY = "H3 自动导演"
    OUTPUT_NODE = True

    @staticmethod
    def _cleanup_after_final():
        """Release model and CUDA allocations only after final output is durable."""
        try:
            model_management.unload_all_models()
        except Exception:
            try:
                model_management.cleanup_models_gc()
            except Exception:
                pass
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @staticmethod
    def _assemble(plan, output_name="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量", video_vae=None, audio_vae=None):
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("未找到 ffmpeg，无法拼接 H3 片段。请重启 ComfyUI，或设置环境变量 FFMPEG_PATH 指向 ffmpeg.exe。")
        project_dir = Path(plan["project_dir"])
        final_dir = project_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        fmt = str(video_format or "mp4").lower().lstrip(".")
        if fmt not in VIDEO_FORMATS:
            fmt = "mp4"
        codec = str(video_codec or "h264").lower()
        if codec not in VIDEO_CODECS:
            codec = "h264"
        final_path = final_dir / ("%s.%s" % (_output_filename(output_name), fmt))
        list_path = project_dir / "json" / "concat.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        assembly_dir = project_dir / "json" / ".assembly_segments"
        assembly_dir.mkdir(parents=True, exist_ok=True)
        state = _load_json(_state_path(plan), {"segments": {}})
        for index in range(1, len(plan.get("segments", [])) + 1):
            # The controller may run in a later queued task, after the sampler
            # has released its tensors.  Always resolve the numbered clip and
            # final AV latent from the project directory on disk; never rely
            # on an in-memory output from the last segment.
            clip, latent_path = _paths(plan, index, output_name)
            deferred_decode = bool(plan.get("decode_after_all_segments", False))
            if not deferred_decode and not clip.is_file():
                raise FileNotFoundError("Cannot assemble; missing segment video: %s" % clip)
            if not latent_path.is_file():
                raise FileNotFoundError("Cannot assemble; missing final AV latent cache for segment %d: %s" % (index, latent_path))
            entry = (state.get("segments") or {}).get(str(index), {})
            # SaveSegment owns context cropping. In deferred mode it records
            # the chosen value here and final assembly applies exactly that
            # recorded value while decoding each latent; no second policy or
            # inference is permitted in the controller.
            trim = max(0, int(entry.get("context_trim_frames", 0) or 0))
            if deferred_decode:
                if video_vae is None or audio_vae is None:
                    raise RuntimeError("开启“所有片段采样完成后统一解码”时，拼接节点必须连接视频 VAE 和音频 VAE")
                parts = _av_latent_parts(_load_av_latent(latent_path))
                if parts is None:
                    raise ValueError("第 %d 段缓存不是 H3 联合 AV latent" % index)
                decoded_images = _decode_h3_video(video_vae, parts[0])
                decoded_audio, sample_rate = _decode_h3_audio(audio_vae, parts[1])
                audio_data = {"waveform": decoded_audio, "sample_rate": sample_rate}
                if trim > 0:
                    decoded_images, audio_data = _trim_context_prefix(decoded_images, audio_data, trim, FPS)
                source = assembly_dir / ("H3_%05d%s" % (index, "." + str(video_format).lower().lstrip(".")))
                _write_segment_video(source, decoded_images, audio_data, FPS, video_format, video_codec, encoder_device, quality)
                LOG.info("H3 Auto Director: 统一解码逐段处理第 %d 段（裁剪 %d 帧）", index, trim)
                del decoded_images, decoded_audio, audio_data, parts
                if torch.cuda.is_available():
                    try: model_management.soft_empty_cache()
                    except Exception: pass
            else:
                _verify_video_stream(clip)
                LOG.info("H3 Auto Director: 从磁盘读取第 %d 段视频与最终 latent：%s / %s", index, clip, latent_path)
                source = clip
            lines.append("file '%s'" % str(source).replace("'", "'\\''"))
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp = final_path.with_name(final_path.stem + ".final_tmp" + final_path.suffix)
        try:
            _encode_concat_with_fallback(ffmpeg, list_path, tmp, fmt, codec, encoder_device, quality)
        finally:
            # Temporary decoded/cropped segment files are disposable and must
            # never be mistaken for resumable project cache.
            shutil.rmtree(assembly_dir, ignore_errors=True)
        # A transfer project can replace all generated segment soundtracks
        # with the original reference video's complete audio stream. Do this
        # after video concatenation so segment padding cannot duplicate or
        # re-encode the source audio at every boundary.
        if (str(plan.get("mode", "")) == "video_transfer"
                and str(plan.get("final_audio_source", "H3 生成音频")) == "参考视频音频"):
            reference = plan.get("reference_video") or {}
            source_name = reference.get("path") or reference.get("name")
            source = (_input_root() / _reference_name(source_name)).resolve()
            if not source.is_file():
                tmp.unlink(missing_ok=True)
                raise FileNotFoundError("参考视频音频源不存在：%s" % source)
            muxed = final_path.with_name(final_path.stem + ".audio_tmp" + final_path.suffix)
            audio_codec = "libopus" if fmt == "webm" else "aac"
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                       "-i", str(tmp), "-i", str(source),
                       "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", audio_codec]
            if fmt in {"mp4", "mov"}:
                command.extend(["-movflags", "+faststart"])
            command.extend(["-f", VIDEO_FORMATS.get(fmt, "mp4"), str(muxed)])
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            except (OSError, subprocess.SubprocessError) as exc:
                result = None
                detail = str(exc)
            else:
                detail = (result.stderr or "")[-1600:]
            tmp.unlink(missing_ok=True)
            if result is None or result.returncode != 0 or not muxed.is_file() or muxed.stat().st_size == 0:
                muxed.unlink(missing_ok=True)
                raise RuntimeError("无法封装参考视频完整音频：%s" % detail)
            os.replace(muxed, tmp)
        os.replace(tmp, final_path)
        return str(final_path)

    def advance(self, plan, segment_index, saved_video, segment_node_id, output_root="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量", video_vae=None, audio_vae=None, cleanup_after_final=True, prompt=None, client_id=None):
        total = len(plan.get("segments", []))
        runtime = dict(plan)
        # ``plan.project_dir`` is authoritative.  In particular, TTS segments
        # live under audio/segments, where parent.parent is the audio folder,
        # not the project directory.
        project_dir = Path(runtime["project_dir"])
        state_path = _state_path(runtime)
        state = _load_json(state_path, {"version": 1, "segments": {}})
        mode = str(runtime.get("mode", ""))
        saved_key = "audio" if mode == "tts" else "video"
        # Preserve SaveSegment's durable paths (final latent, raw context
        # video, crop count) while recording the controller output.
        segment_state = state.setdefault("segments", {}).setdefault(str(int(segment_index)), {})
        segment_state.update({"status": "completed", saved_key: saved_video})
        state["last_completed"] = int(segment_index)
        _atomic_json(state_path, state)
        if int(segment_index) >= total or not bool(plan.get("auto_run", True)):
            final_path = ""
            if int(segment_index) >= total:
                if str(plan.get("mode", "")) == "tts" and not bool(plan.get("concat_final_audio", True)):
                    final_path = ""
                    state["final_audio"] = ""
                else:
                    final_path = self._assemble(runtime, output_root, video_format, video_codec, encoder_device, quality,
                                                 video_vae=video_vae, audio_vae=audio_vae)
                    final_file = Path(final_path)
                    if not final_file.is_file() or final_file.stat().st_size <= 0:
                        raise RuntimeError("最终输出拼接返回了空文件，已保留显存与运行状态供排查")
                    state["final_video" if str(plan.get("mode", "")) != "tts" else "final_audio"] = final_path
                state["status"] = "complete"
                state.pop("next_segment", None)
                if cleanup_after_final:
                    self._cleanup_after_final()
            else:
                state["status"] = "paused"
            _atomic_json(state_path, state)
            return (state["status"], final_path)
        if not prompt or not segment_node_id or str(segment_node_id) not in prompt:
            raise ValueError("Set segment_node_id to the H3AutoDirectorSegment node id")
        next_prompt = copy.deepcopy(prompt)
        next_prompt[str(segment_node_id)].setdefault("inputs", {})["segment_index"] = int(segment_index)
        # Carry the resolved project directory into the queued plan so the
        # next run cannot rebuild a different path from legacy field values.
        plan_node_types = {
            "H3AutoDirectorPlan",
            "H3AutoDirectorTTSPlan",
            "H3AutoDirectorVideoTransferPlan",
        }
        for data in next_prompt.values():
            if data.get("class_type") in plan_node_types:
                data.setdefault("inputs", {})["project_dir"] = str(project_dir)
        output_nodes = []
        for node_id, data in next_prompt.items():
            cls = nodes.NODE_CLASS_MAPPINGS.get(data.get("class_type"))
            if cls is not None and getattr(cls, "OUTPUT_NODE", False):
                output_nodes.append(node_id)
        from server import PromptServer
        server = PromptServer.instance
        prompt_id = str(uuid.uuid4())
        number = server.number
        server.number += 1
        extra = {"client_id": client_id} if client_id else {}
        try:
            server.prompt_queue.put((number, prompt_id, next_prompt, extra, output_nodes, {}))
        except Exception as exc:
            state["status"] = "queue_failed"
            state["queue_error"] = str(exc)
            _atomic_json(state_path, state)
            raise RuntimeError("自动续跑任务入队失败：%s" % exc) from exc
        state["status"] = "queued"
        state["next_segment"] = int(segment_index) + 1
        _atomic_json(state_path, state)
        LOG.info("H3 Auto Director: 已保存第 %d/%d 段，已自动排队第 %d 段", int(segment_index), total, int(segment_index) + 1)
        return ("queued segment %d/%d" % (int(segment_index) + 1, total),)


class H3AutoDirectorTTSController(H3AutoDirectorController):
    """Reuse project queueing while concatenating numbered WAV segments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("H3_AUTO_PLAN",), "segment_index": ("INT", {"default": 1, "min": 1}),
            "saved_video": ("STRING", {"forceInput": True}),
            "segment_node_id": ("STRING", {"forceInput": True}),
            "output_root": ("STRING", {"default": "", "tooltip": "最终长 WAV 文件名；留空使用 H3"}),
        }, "optional": {
            "cleanup_after_final": ("BOOLEAN", {"default": True}),
        }, "hidden": {"prompt": "PROMPT", "client_id": "CLIENT_ID"}}

    RETURN_NAMES = ("运行状态", "最终音频")
    CATEGORY = "H3 自动导演/TTS"

    @staticmethod
    def _assemble(plan, output_name="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量", **_unused):
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("未找到 ffmpeg，无法拼接 TTS 音频")
        project_dir = Path(plan["project_dir"])
        final_dir = project_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / ("%s.wav" % _output_filename(output_name))
        list_path = project_dir / "json" / "audio_concat.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for index in range(1, len(plan.get("segments", [])) + 1):
            path = _audio_path(plan, index, _segment(plan, index).get("audio_filename", ""))
            if not path.is_file():
                raise FileNotFoundError("缺少 TTS 片段音频：%s" % path)
            lines.append("file '%s'" % str(path).replace("'", "'\\''"))
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp = final_path.with_name(final_path.stem + ".tmp.wav")
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-vn", "-c:a", "pcm_s16le", str(tmp)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if result.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("TTS 长音频拼接失败：%s" % (result.stderr or "")[-1600:])
        os.replace(tmp, final_path)
        return str(final_path)


class H3AutoDirectorSamplingSwitch:
    """Provide H3 audio-sampling metadata and a step-free SIGMAS base."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampling_mode": ([NATIVE_MODE, LEGACY_MODE], {"default": NATIVE_MODE}),
            "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
        }, "optional": {
            "model": ("MODEL", {"tooltip": "可选。连接后，SIGMAS 输出会使用已应用视频/音频偏移的模型端点；未连接时输出安全的默认端点。"}),
        }}

    RETURN_TYPES = ("H3_AUDIO_SAMPLING", "SIGMAS")
    RETURN_NAMES = ("采样调度信息", "SIGMAS")
    FUNCTION = "apply"
    CATEGORY = "H3 自动导演/音频采样"

    def apply(self, sampling_mode, shift_video, shift_audio, model=None, **_legacy_unused):
        video_shift = float(shift_video)
        audio_shift = float(shift_audio)
        sampling_info = {
            "sampling_mode": str(sampling_mode),
            "shift_video": video_shift,
            "shift_audio": audio_shift,
        }
        # This output deliberately contains only the two endpoints.  It is a
        # valid SIGMAS tensor for ExtendIntermediateSigmas, whose own `steps`
        # control determines the final number of sampling steps.  Do not call
        # _h3_sigmas() here: doing so would make this node silently own a step
        # count and override the user's sampler settings.
        sigma_max = 1.0
        if model is not None:
            try:
                patched = apply_h3_sampling(model, sampling_info["sampling_mode"], video_shift, audio_shift)
                model_sampling = patched.get_model_object("model_sampling")
                sigma_max = float(model_sampling.sigma_max)
            except Exception as exc:
                LOG.warning("H3 Auto Director: 无法读取应用音频采样配置后的 sigma_max，SIGMAS 使用默认端点：%s", exc)
                sigma_max = 1.0
        if not math.isfinite(sigma_max) or sigma_max <= 0.0:
            LOG.warning("H3 Auto Director: 模型 sigma_max 无效，SIGMAS 使用默认端点")
            sigma_max = 1.0
        base_sigmas = torch.tensor([sigma_max, 0.0], dtype=torch.float32)
        setattr(base_sigmas, _AUDIO_SAMPLING_SIGMAS_MARKER, True)
        setattr(base_sigmas, _AUDIO_SAMPLING_SIGMAS_INFO, dict(sampling_info))
        return (sampling_info, base_sigmas)


class H3AutoDirectorApplyAudioSampling:
    """Apply an audio-sampling configuration to a model for standard samplers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "audio_sampling": ("H3_AUDIO_SAMPLING",),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("模型",)
    FUNCTION = "apply"
    CATEGORY = "H3 自动导演/音频采样"

    def apply(self, model, audio_sampling):
        return (_apply_audio_sampling_config(model, audio_sampling, "模型"),)


NODE_CLASS_MAPPINGS = {
    "H3AutoDirectorPlan": H3AutoDirectorPlan,
    "H3AutoDirectorTTSPlan": H3AutoDirectorTTSPlan,
    "H3AutoDirectorVideoTransferPlan": H3AutoDirectorVideoTransferPlan,
    "H3AutoDirectorControlPreprocess": H3AutoDirectorControlPreprocess,
    "H3AutoDirectorControlConfig": H3AutoDirectorControlConfig,
    "H3AutoDirectorControlExport": H3AutoDirectorControlExport,
    "H3AutoDirectorControlBackendCheck": H3AutoDirectorControlBackendCheck,
    "H3AutoDirectorHybridModelLoader": H3AutoDirectorHybridModelLoader,
    "H3AutoDirectorTransferModelLoader": H3AutoDirectorTransferModelLoader,
    "H3AutoDirectorDualStageModelLoader": H3AutoDirectorDualStageModelLoader,
    "H3AutoDirectorSegment": H3AutoDirectorSegment,
    "H3AutoDirectorReferenceResolver": H3AutoDirectorReferenceResolver,
    "H3AutoDirectorCachedReferenceToVideo": H3AutoDirectorCachedReferenceToVideo,
    "H3AutoDirectorResolution": H3AutoDirectorResolution,
    "H3AutoDirectorDecodeSaveVideo": H3AutoDirectorDecodeSaveVideo,
    "H3AutoDirectorLoadSavedAVLatent": H3AutoDirectorLoadSavedAVLatent,
    "H3AutoDirectorDualSampling": H3AutoDirectorDualSamplingModel,
    "H3AutoDirectorAVDecode": H3AutoDirectorAVDecode,
    "H3AutoDirectorTransferDecode": H3AutoDirectorTransferDecode,
    "H3AutoDirectorAudioDecode": H3AutoDirectorAudioDecode,
    "H3AutoDirectorSaveAudioSegment": H3AutoDirectorSaveAudioSegment,
    "H3AutoDirectorContext": H3AutoDirectorContext,
    "H3AutoDirectorResumeContext": H3AutoDirectorResumeContext,
    "H3AutoDirectorMotionContext": H3AutoDirectorMotionContext,
    "H3AutoDirectorSaveSegment": H3AutoDirectorSaveSegment,
    "H3AutoDirectorController": H3AutoDirectorController,
    "H3AutoDirectorTTSController": H3AutoDirectorTTSController,
    "H3AutoDirectorSamplingSwitch": H3AutoDirectorSamplingSwitch,
    "H3AutoDirectorApplyAudioSampling": H3AutoDirectorApplyAudioSampling,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AutoDirectorPlan": "H3 自动导演｜项目计划",
    "H3AutoDirectorTTSPlan": "H3 自动导演｜TTS 项目计划",
    "H3AutoDirectorVideoTransferPlan": "H3 自动导演｜动作迁移项目计划",
    "H3AutoDirectorControlPreprocess": "H3 动作迁移｜姿态/深度预处理",
    "H3AutoDirectorControlConfig": "H3 动作迁移｜Union 控制配置",
    "H3AutoDirectorControlExport": "H3 动作迁移｜导出 Union 控制视频",
    "H3AutoDirectorControlBackendCheck": "H3 动作迁移｜Union 后端检查",
    "H3AutoDirectorHybridModelLoader": "H3 自动导演｜多模态参考模型加载",
    "H3AutoDirectorTransferModelLoader": "H3 自动导演｜动作迁移模型加载",
    "H3AutoDirectorDualStageModelLoader": "H3 自动导演｜一采/二采模型加载",
    "H3AutoDirectorSegment": "H3 自动导演｜片段设置",
    "H3AutoDirectorReferenceResolver": "H3 自动导演｜多模态素材解析",
    "H3AutoDirectorCachedReferenceToVideo": "H3 自动导演｜提示词与素材缓存",
    "H3AutoDirectorResolution": "H3 自动导演｜双采样分辨率",
    "H3AutoDirectorDecodeSaveVideo": "H3 解码｜目录批量解码并保存视频",
    "H3AutoDirectorLoadSavedAVLatent": "H3 解码｜加载保存的 AV 潜空间",
    "H3AutoDirectorDualSampling": "H3 自动导演｜双阶段采样",
    "H3AutoDirectorAVDecode": "H3 自动导演｜AV 解码",
    "H3AutoDirectorTransferDecode": "H3 视频迁移｜按策略解码",
    "H3AutoDirectorAudioDecode": "H3 TTS｜仅解码音频",
    "H3AutoDirectorSaveAudioSegment": "H3 TTS｜保存音频片段",
    "H3AutoDirectorContext": "H3 自动导演｜上下文读取",
    "H3AutoDirectorResumeContext": "H3 自动导演｜断点续接",
    "H3AutoDirectorMotionContext": "H3 自动导演｜运动上下文",
    "H3AutoDirectorSaveSegment": "H3 自动导演｜保存片段",
    "H3AutoDirectorController": "H3 自动导演｜拼接最终视频",
    "H3AutoDirectorTTSController": "H3 TTS｜拼接最终音频",
    "H3AutoDirectorSamplingSwitch": "H3 自动导演｜音频采样切换",
    "H3AutoDirectorApplyAudioSampling": "H3 自动导演｜应用音频采样配置",
}
