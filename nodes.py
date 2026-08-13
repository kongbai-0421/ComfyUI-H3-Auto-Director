"""Project runner primitives for MiniMax H3 in ComfyUI.

The controller queues the next copy of the current workflow after a segment
has been saved. It deliberately uses numbered project slots, so a rerun never
silently consumes the newest rejected cache.
"""

from __future__ import annotations

import copy
import importlib
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
from .sampling_switch import LEGACY_MODE, NATIVE_MODE, apply_h3_sampling

try:
    from comfy_extras import nodes_minimax_h3 as _minimax_h3
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo as _H3ReferenceToVideo
except ImportError:
    _minimax_h3 = None
    _H3ReferenceToVideo = None

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
FRAME_CONTEXT_DEFAULT = 22
PROMPT_CACHE_MAX_PROJECTS = 2
_PROMPT_CONDITIONING_CACHE = OrderedDict()
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCE_TOTAL = 12
PROJECT_ROOT_NAME = "h3_project"

VIDEO_FORMATS = {"mp4": "mp4", "mkv": "matroska", "webm": "webm", "mov": "mov"}
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov")
VIDEO_CODECS = {
    "h264": {"cpu": "libx264", "gpu": ("h264_nvenc", "h264_qsv", "h264_amf")},
    "hevc": {"cpu": "libx265", "gpu": ("hevc_nvenc", "hevc_qsv", "hevc_amf")},
    "vp9": {"cpu": "libvpx-vp9", "gpu": ()},
    "av1": {"cpu": "libaom-av1", "gpu": ("av1_nvenc", "av1_qsv", "av1_amf")},
}
QUALITY_CHOICES = ("最高质量", "高质量", "平衡", "快速")
ENCODER_DEVICES = ("CPU", "GPU")
COLOR_CORRECTION_CHOICES = ("关闭", "匹配首段", "匹配上段")
CONTEXT_DIR_NAME = "context"
DUAL_UPSCALE_CHOICES = ("普通插值", "普通放大模型", "RTX Video Super Resolution", "自动（RTX→普通模型→插值）")
_MOTION_CONTEXT_MARKER = "_h3_auto_director_motion_context"


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


def _align_frames(frames: int) -> int:
    frames = max(5, int(frames))
    while frames % 17 != 5:
        frames += 1
    return frames


def _align_frames_nearest(frames: int) -> int:
    """Use the closest valid H3 duration instead of always rounding upward.

    With video context the pinned head is removed after decoding.  Rounding
    the *pre-trim* length up can add nearly a full 17-frame H3 block to the
    delivered clip.  Selecting the nearest valid physical duration preserves
    the requested visible duration as closely as the model's grid permits.
    """
    target = max(5, int(frames))
    lower = target - ((target - 5) % 17)
    lower = max(5, lower)
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
    return bool(plan.get("video_continuation", plan.get("continuation_mode", True)))


def _use_previous_video_reference(plan, generation_index: int) -> bool:
    """Read the previous-video reference flag from the target segment only."""
    if int(generation_index) <= 1:
        return False
    seg = _segment(plan, int(generation_index))
    return bool(seg.get("use_previous_video_reference", False))


def _segment_reference_specs(plan, generation_index: int):
    """Return user references plus a runtime-only previous-video reference when enabled."""
    seg = _segment(plan, int(generation_index))
    refs = list(plan.get("global_assets", [])) if plan.get("global_reference_set", True) else list(seg.get("references", []))
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


def _paths(plan, index: int, output_name="", video_format="mp4", for_write=False, for_context=False):
    base = Path(plan["project_dir"])
    clips = base / (CONTEXT_DIR_NAME if for_context else "clips")
    cache = base / "cache"
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
    tail = deque(maxlen=max_frames)
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


def _dual_sample(model, conditioning, latent, sampler_name, scheduler, steps, denoise, seed):
    """Run one positive-only H3 sampling pass, matching BasicGuider semantics."""
    if not conditioning:
        raise ValueError("双采样需要有效的正向条件")
    from comfy_extras.nodes_custom_sampler import Guider_Basic
    guider = Guider_Basic(model)
    guider.set_conds(conditioning)
    sampler = comfy.samplers.sampler_object(str(sampler_name))
    sigmas = _h3_sigmas(model, scheduler, steps, denoise)
    if sigmas.numel() == 0:
        return dict(latent)
    latent_image = latent["samples"]
    noise = comfy.sample.prepare_noise(latent_image, int(seed), latent.get("batch_index"))
    samples = guider.sample(noise, latent_image, sampler, sigmas,
                            denoise_mask=latent.get("noise_mask"), seed=int(seed))
    result = dict(latent)
    result["samples"] = samples
    return result


def _mark_motion_context(conditioning):
    """Tag only the conditioning emitted by this adapter, not user references."""
    return node_helpers.conditioning_set_values(conditioning, {_MOTION_CONTEXT_MARKER: True})


def _prepare_dual_sampling_conditioning(conditioning, strip_motion_context=False):
    """Remove internal tags and, for stage two, only Motion Context payloads.

    The stage-one sampler is responsible for joining a project segment to its
    predecessor.  Stage two starts from the upscaled stage-one latent, so
    pinning the predecessor a second time competes with that latent and
    effectively applies continuation twice.  User supplied ``minimax_refs``
    must remain available in both stages.
    """
    prepared = []
    for entry in conditioning or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            prepared.append(entry)
            continue
        values = entry[1].copy()
        is_motion_context = bool(values.pop(_MOTION_CONTEXT_MARKER, False))
        if strip_motion_context and is_motion_context:
            values.pop("minimax_keyframes", None)
            values.pop("minimax_frame_count", None)
            refs = values.get("minimax_refs")
            if isinstance(refs, (list, tuple)):
                # The adapter marks its audio continuation ref with the
                # Motion Context timeline field. Leave all normal user refs.
                retained = [ref for ref in refs if not (
                    isinstance(ref, dict) and "motion_context_audio_end_frame" in ref
                )]
                if retained:
                    values["minimax_refs"] = retained
                else:
                    values.pop("minimax_refs", None)
        prepared.append([entry[0], values, *entry[2:]])
    return prepared


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
    with nvvfx.VideoSuperRes(quality_map.get(str(quality), nvvfx.effects.QualityLevel.HIGH)) as sr:
        sr.output_width, sr.output_height = output_width, output_height
        sr.load()
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
    return restore(out)


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
            "multiple": ("INT", {"default": 32, "min": 16, "max": 128, "step": 16}),
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
        multiple = max(1, int(multiple))
        width = max(multiple, int(round(width / multiple)) * multiple)
        height = max(multiple, int(round(height / multiple)) * multiple)
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
            "model": ("MODEL",), "conditioning": ("CONDITIONING",), "latent": ("LATENT",),
            "video_vae": ("VAE",), "audio_vae": ("VAE",),
            "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "res_multistep"}),
            "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
            "stage1_steps": ("INT", {"default": 6, "min": 1, "max": 100}),
            "stage1_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_stage2": ("BOOLEAN", {"default": True, "label_on": "启用二采", "label_off": "关闭二采",
                               "tooltip": "关闭后仅执行第一阶段采样，跳过放大、视频 VAE 重编码和第二阶段采样。"}),
            "stage2_use_context": ("BOOLEAN", {"default": False, "label_on": "二采使用上下文", "label_off": "二采不使用上下文",
                                    "tooltip": "开启后将第一阶段的项目视频/音频上下文也传入第二阶段；默认关闭，避免上下文被重复施加。"}),
            "stage2_steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "stage2_denoise": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            "upscale_mode": (DUAL_UPSCALE_CHOICES, {"default": "普通插值"}),
            "target_width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32, "forceInput": True}),
            "target_height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32, "forceInput": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }, "optional": {
            "upscale_model": ("UPSCALE_MODEL",),
            "stage2_conditioning": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE")
    RETURN_NAMES = ("最终 AV latent", "第一阶段 AV latent", "放大预览")
    FUNCTION = "sample"
    CATEGORY = "H3 自动导演/采样"

    def sample(self, model, conditioning, latent, video_vae, audio_vae, sampler_name, scheduler,
               stage1_steps, stage1_denoise, stage2_steps, stage2_denoise, upscale_mode,
               target_width, target_height, enable_stage2=True, stage2_use_context=False,
               upscale_model=None, seed=0, stage2_conditioning=None):
        stage1_conditioning = _prepare_dual_sampling_conditioning(conditioning)
        first = _dual_sample(model, stage1_conditioning, latent, sampler_name, scheduler, stage1_steps, stage1_denoise, seed)
        if not bool(enable_stage2):
            # Preserve the output contract without paying for decode/upscale.
            # The empty IMAGE is deliberately inert; the final AV latent is
            # the first-stage result and remains compatible with AV Decode.
            empty_preview = torch.empty((0, 1, 1, 3), dtype=torch.float32)
            return (first, first, empty_preview)
        parts = _av_latent_parts(first)
        if parts is None:
            raise ValueError("双采样仅支持 MiniMax H3 联合 AV latent")
        first_video, first_audio = parts
        decoded = _decode_h3_video(video_vae, first_video)
        expected_frames = _expected_decoded_frames(video_vae, first_video)
        width = max(32, int(target_width) // 32 * 32)
        height = max(32, int(target_height) // 32 * 32)
        mode = str(upscale_mode)
        if mode == "RTX Video Super Resolution":
            preview = _upscale_rtx(decoded, width, height, "高")
        elif mode == "普通放大模型":
            preview = _upscale_with_model(upscale_model, decoded)
            preview = _upscale_interpolate(preview, width, height)
        elif mode == "自动（RTX→普通模型→插值）":
            try:
                preview = _upscale_rtx(decoded, width, height, "高")
            except Exception:
                preview = _upscale_with_model(upscale_model, decoded) if upscale_model else decoded
                preview = _upscale_interpolate(preview, width, height)
        else:
            preview = _upscale_interpolate(decoded, width, height)
        # Some VAE/upscaler combinations round the temporal dimension while
        # processing a video.  Restore H3's exact 17k+5 frame count before the
        # second VAE encode; spatial scaling must never change time.
        if expected_frames is not None:
            preview = _match_video_frame_count(preview, expected_frames)
        encoded_video = _encode_h3_video(video_vae, preview)
        encoded_video = _match_latent_time(encoded_video, first_video.shape[2])
        import comfy.nested_tensor
        # The two branches can be returned on different devices/dtypes by
        # custom VAEs.  Normalize audio before rebuilding H3's AV container.
        first_audio = first_audio.to(device=encoded_video.device, dtype=encoded_video.dtype)
        refined = dict(first)
        refined["samples"] = comfy.nested_tensor.NestedTensor((encoded_video, first_audio))
        # Retain text and user references, but remove the project continuation
        # injected for stage one unless the user explicitly asks to apply it
        # again during refinement. This also applies to a separately connected
        # stage-two conditioning if it came from our Motion Context adapter.
        final_conditioning = _prepare_dual_sampling_conditioning(
            stage2_conditioning if stage2_conditioning is not None else conditioning,
            strip_motion_context=not bool(stage2_use_context),
        )
        final = _dual_sample(model, final_conditioning, refined, sampler_name, scheduler, stage2_steps, stage2_denoise, int(seed) + 1)
        return (final, first, preview)


class H3AutoDirectorDualSamplingModel:
    """Graph-friendly dual sampler that accepts the existing H3 model chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "conditioning": ("CONDITIONING",), "latent": ("LATENT",),
            "video_vae": ("VAE",), "audio_vae": ("VAE",),
            "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "res_multistep"}),
            "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
            "stage1_steps": ("INT", {"default": 6, "min": 1, "max": 100}),
            "stage1_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "enable_stage2": ("BOOLEAN", {"default": True, "label_on": "启用二采", "label_off": "关闭二采",
                               "tooltip": "关闭后仅执行第一阶段采样。"}),
            "stage2_use_context": ("BOOLEAN", {"default": False, "label_on": "二采使用上下文", "label_off": "二采不使用上下文",
                                    "tooltip": "开启后第二阶段也使用视频/音频上下文接续。"}),
            "stage2_steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "stage2_denoise": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            "upscale_mode": (DUAL_UPSCALE_CHOICES, {"default": "普通插值"}),
            "target_width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32, "forceInput": True}),
            "target_height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32, "forceInput": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }, "optional": {"upscale_model": ("UPSCALE_MODEL",), "stage2_conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE")
    RETURN_NAMES = ("最终 AV latent", "第一阶段 AV latent", "放大预览")
    FUNCTION = "sample"
    CATEGORY = "H3 自动导演/采样"

    def sample(self, model, conditioning, latent, video_vae, audio_vae, sampler_name, scheduler,
        stage1_steps, stage1_denoise, stage2_steps, stage2_denoise, upscale_mode,
               target_width, target_height, enable_stage2=True, stage2_use_context=False,
               seed=0, upscale_model=None, stage2_conditioning=None):
        return H3AutoDirectorDualSampling().sample(
        model, conditioning, latent, video_vae, audio_vae, sampler_name, scheduler,
            stage1_steps, stage1_denoise, stage2_steps, stage2_denoise, upscale_mode,
            target_width, target_height, enable_stage2, stage2_use_context, upscale_model, seed, stage2_conditioning)


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
            "output_root": ("STRING", {"default": "h3_projects", "tooltip": "项目文件夹名称；新路径为 output/h3_project/<此名称>"}),
        }, "optional": {
            "global_assets_json": ("STRING", {"default": "[]", "multiline": True}),
        }, "hidden": {"project_dir": "STRING"}}

    RETURN_TYPES = ("H3_AUTO_PLAN",)
    RETURN_NAMES = ("项目计划",)
    FUNCTION = "create"
    CATEGORY = "H3 自动导演"

    def create(self, project_id, segments_json, duration, global_reference_set, auto_run, continuation_mode=True, cache_prompt_embeddings=False, output_root="h3_projects", global_assets_json="[]", project_dir="", **_legacy_inputs):
        try:
            segments = json.loads(segments_json)
            assets = json.loads(global_assets_json or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError("segments_json/global_assets_json must be valid JSON: %s" % exc) from exc
        if not isinstance(segments, list) or not segments:
            raise ValueError("At least one H3 segment is required")
        if not isinstance(assets, list):
            raise ValueError("global_assets_json must be a JSON list")
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
            row["references"] = list(row.get("references", []))
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
                "cache_prompt_embeddings": bool(cache_prompt_embeddings), "global_assets": assets, "segments": normalized,
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
               auto_run=True, output_root="h3_video_transfer", project_dir=""):
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
        return (model, clip, video_vae, audio_vae)


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
        physical = (_align_frames_nearest(target + int(context_length))
                    if use_video else _align_frames(target))
        refs = _segment_reference_specs(plan, generation_index)
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
    loader = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideoPath")
    if loader is None:
        loader = nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideo")
    if loader is None:
        raise RuntimeError("需要安装 VideoHelperSuite 才能加载视频参考素材")
    result = loader().load_video(video=str((_input_root() / clean).resolve()), force_rate=24,
                                 custom_width=0, custom_height=0, frame_load_cap=0,
                                 skip_first_frames=0, select_every_nth=1)
    return result[0], result[2]


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
    result = loader().load_audio(
        audio_file=str((_input_root() / clean).resolve()), seek_seconds=start_seconds,
        duration=duration_seconds)
    return result[0]


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
    use_video = (not _use_previous_video_reference(plan, generation_index)
                 and _video_context_enabled(plan)
                 and bool(seg.get("continue_video", generation_index > 1)) and generation_index > 1)
    target = round(float(seg["duration"]) * FPS)
    return (_align_frames_nearest(target + int(context_length))
            if use_video else _align_frames(target))


def _prompt_cache_key(plan, clip, vae, audio_vae, width, height, ref_image_size, context_length):
    # The seed belongs to RandomNoise/sampling downstream.  It is deliberately
    # absent here so changing the seed reuses the deterministic H3 conditioning.
    plan_data = {k: plan.get(k) for k in ("project_id", "global_reference_set", "global_assets", "segments", "continuation_mode")}
    return (id(clip), id(vae), id(audio_vae), int(width), int(height), str(ref_image_size), int(context_length),
            json.dumps(plan_data, ensure_ascii=False, sort_keys=True, default=str))


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
    def _encode_one(clip, vae, audio_vae, prompt, width, height, length, ref_image_size, refs, plan=None):
        if _H3ReferenceToVideo is None:
            raise RuntimeError("当前 ComfyUI 未提供 MiniMaxH3ReferenceToVideo 核心节点")
        ref_groups = _resolve_reference_groups(refs, plan=plan)
        result = _H3ReferenceToVideo.execute(
            clip, vae, audio_vae, prompt, int(width), int(height), int(length), str(ref_image_size),
            ref_images=ref_groups[0], ref_videos=ref_groups[1],
            ref_video_audios=ref_groups[2], ref_audios=ref_groups[3])
        return result[0], result[1]

    @staticmethod
    def _prepare_references(vae, audio_vae, width, height, length, ref_image_size, refs, plan=None):
        """Encode all Ref2VA assets before the batch text-encoder session."""
        if _H3ReferenceToVideo is None or _minimax_h3 is None:
            raise RuntimeError("当前 ComfyUI 未提供 MiniMaxH3ReferenceToVideo 核心节点")
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
                audio_latent, audio_length = _H3ReferenceToVideo._encode_ref_audio(audio_vae, soundtrack)
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
            audio_latent, audio_length = _H3ReferenceToVideo._encode_ref_audio(audio_vae, audio)
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
    def _build_cache(cls, plan, clip, vae, audio_vae, width, height, ref_image_size, context_length):
        prepared = {}
        pending = []
        segment_count = len(plan.get("segments", []))
        LOG.info("H3 Auto Director: 正在预编码 %d 段参考素材，随后将连续缓存当前可用的文本向量", segment_count)
        for generation_index in range(1, len(plan.get("segments", [])) + 1):
            length = _cache_frame_count(plan, generation_index, context_length)
            refs = _cache_segment_references(plan, generation_index)
            if not cls._references_ready(plan, refs):
                pending.append(generation_index)
                continue
            prepared[generation_index] = cls._prepare_references(
                vae, audio_vae, width, height, length, ref_image_size, refs, plan=plan)

        LOG.info("H3 Auto Director: 参考素材预编码完成，开始连续缓存 %d 段文本向量", len(prepared))
        cache = {}
        for generation_index, prepared_segment in prepared.items():
            seg = _segment(plan, generation_index)
            refs = _cache_segment_references(plan, generation_index)
            cache[generation_index] = cls._encode_prepared_prompt(
                clip, _previous_video_prompt(seg.get("prompt", ""), refs), *prepared_segment)
            LOG.info(
                "H3 Auto Director: 已完成提示词向量缓存 %d/%d（第 %d 段）",
                len(cache), segment_count, generation_index,
            )
        if pending:
            LOG.info("H3 Auto Director: %d 段等待上片段视频生成，暂不卸载文本编码器；生成后按段补齐向量", len(pending))
        else:
            model_management.unload_model_and_clones(clip.patcher, unload_additional_models=False, all_devices=True)
            LOG.info("H3 Auto Director: 全部文本向量缓存完成，已卸载文本编码器")
        return cache

    @classmethod
    def encode(cls, plan, clip, vae, audio_vae, prompt, width, height, length,
               ref_image_size="match", context_length=FRAME_CONTEXT_DEFAULT, segment_index=0,
               references_json=None):
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
        if not bool(plan.get("cache_prompt_embeddings", False)):
            return cls._encode_one(clip, vae, audio_vae, prompt, width, height, length,
                                   ref_image_size, _cache_segment_references(plan, generation_index) if refs is None else refs,
                                   plan=plan)
        key = _prompt_cache_key(plan, clip, vae, audio_vae, width, height, ref_image_size, context_length)
        cache = _PROMPT_CONDITIONING_CACHE.get(key)
        if cache is None:
            cache = cls._build_cache(plan, clip, vae, audio_vae, width, height, ref_image_size, context_length)
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
                width, height, length, ref_image_size, effective_refs, plan=plan)
            _PROMPT_CONDITIONING_CACHE[key] = cache
            LOG.info(
                "H3 Auto Director: 已完成提示词向量缓存 %d/%d（第 %d 段，延迟参考素材就绪）",
                len(cache), len(plan.get("segments", [])), generation_index,
            )
            if len(cache) >= len(plan.get("segments", [])):
                model_management.unload_model_and_clones(clip.patcher, unload_additional_models=False, all_devices=True)
                LOG.info("H3 Auto Director: 延迟参考素材就绪，全部文本向量缓存完成，已卸载文本编码器")
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
        video, audio = parts
        images = _decode_h3_video(video_vae, video)
        source_audio = str(plan.get("final_audio_source", "H3 生成音频")) == "参考视频音频"
        skip = bool(plan.get("skip_h3_audio_decode", False))
        if source_audio or skip:
            if source_audio:
                return (images, _load_transfer_video_audio(_transfer_segment_ref(plan, int(segment_index) + 1)))
            return (images, None)
        waveform = audio_vae.decode(audio).movedim(-1, 1)
        sample_rate = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 32000))
        return (images, {"waveform": waveform, "sample_rate": sample_rate})


class H3AutoDirectorAVDecode:
    """Decode the final H3 joint AV latent for the standard director plan."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "video_vae": ("VAE",), "audio_vae": ("VAE",),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("画面", "音频")
    FUNCTION = "decode"
    CATEGORY = "H3 自动导演/解码"

    def decode(self, samples, video_vae, audio_vae):
        parts = _av_latent_parts(samples)
        if parts is None:
            raise ValueError("H3 AV 解码需要联合视频/音频 latent")
        images = _decode_h3_video(video_vae, parts[0])
        waveform = audio_vae.decode(parts[1]).movedim(-1, 1)
        sample_rate = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 32000))
        return (images, {"waveform": waveform, "sample_rate": sample_rate})


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
        waveform = audio_vae.decode(parts[1]).movedim(-1, 1)
        sample_rate = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 32000))
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
        return {"required": {"plan": ("H3_AUTO_PLAN",), "segment_index": ("INT", {"default": 0, "min": 0})}}

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("上下文画面", "上下文潜变量")
    FUNCTION = "load"
    CATEGORY = "H3 自动导演"

    def load(self, plan, segment_index):
        video_enabled = _video_context_enabled(plan)
        audio_enabled = bool(plan.get("continuation_mode", True))
        if int(segment_index) <= 0 or not (video_enabled or audio_enabled):
            return (torch.zeros((1, 1, 1, 3), dtype=torch.float32), {"samples": [torch.zeros((1, 24, 2, 1, 1)), torch.zeros((1, 32, 2, 1))]})
        video_path, latent_path = _paths(plan, int(segment_index), for_context=True)
        if not latent_path.exists() or (video_enabled and not video_path.exists()):
            raise FileNotFoundError("Missing context cache for segment %d: %s / %s" % (int(segment_index), video_path, latent_path))
        frames = _load_context_video(video_path) if video_enabled else torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return (frames, _load_av_latent(latent_path))


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
        }}

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("条件", "裁剪帧数")
    FUNCTION = "apply"
    CATEGORY = "H3 自动导演"

    @staticmethod
    def _direct_latent_context(conditioning, latent, context_latent, context_length, use_audio_context,
                               use_video_context=True):
        """Build Motion Context payloads from the cached AV latent without VAE re-encoding."""
        cls = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3MotionContext")
        if cls is None:
            raise RuntimeError("未安装 ComfyUI-H3-Motion-Context")
        try:
            motion_nodes = importlib.import_module(cls.__module__)
            package = cls.__module__.rsplit(".", 1)[0]
            layout_module = importlib.import_module(package + ".patch_layout")
            payload_module = importlib.import_module(package + ".patch_payload")
            MC_KEY = layout_module.MC_KEY
            MC_AUDIO_KEY = layout_module.MC_AUDIO_KEY
            is_applied = layout_module.is_applied
            payload_patch_applied = payload_module.is_applied
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("ComfyUI-H3-Motion-Context 未提供 latent 直取所需接口") from exc
        if not is_applied():
            raise RuntimeError("H3 Motion Context 布局补丁未启用，无法使用视频 latent 直取")
        target_parts = _av_latent_parts(latent)
        context_parts = _av_latent_parts(context_latent)
        if target_parts is None or context_parts is None:
            raise ValueError("上下文 AV latent 无效，无法直接读取视频 latent")
        target_video, context_video = target_parts[0], context_parts[0]
        if not torch.is_tensor(target_video) or not torch.is_tensor(context_video) or target_video.ndim != 5 or context_video.ndim != 5:
            raise ValueError("H3 视频 latent 必须是 [B,C,T,H,W]")
        if target_video.shape[0] != context_video.shape[0] or target_video.shape[1] != context_video.shape[1] or target_video.shape[3:] != context_video.shape[3:]:
            raise ValueError("上下文 latent 分辨率或通道与当前片段不一致")
        run = next((value for value in (39, 22, 5, 1) if value <= int(context_length)), 1)
        steps = {1: 1, 5: 2, 22: 7, 39: 12}[run]
        if not use_video_context:
            if not use_audio_context:
                return conditioning, 0
            if not payload_patch_applied():
                raise RuntimeError("H3 Motion Context payload 补丁未启用，无法安全使用音频上下文")
            audio_tail, audio_steps, overhang = motion_nodes._audio_tail_from_latent(context_latent, run)
            values = {"minimax_refs": [{
                "kind": "audio", "ref_audio_t": audio_steps,
                "audio_latent": audio_tail.to(device=target_video.device, dtype=target_video.dtype),
                # Audio-only continuation starts at the first frame because
                # no video rows are pinned in this mode.
                MC_AUDIO_KEY: 0.0,
            }]}
            return _mark_motion_context(node_helpers.conditioning_set_values(conditioning, values)), 0
        if context_video.shape[2] < steps:
            raise ValueError("上下文视频 latent 长度不足，无法读取 %d 帧上下文" % run)
        if run >= motion_nodes._pixel_frames(int(target_video.shape[2])):
            raise ValueError("上下文长度不能占满当前生成片段")
        tail = context_video[:, :, -steps:].to(device=target_video.device, dtype=target_video.dtype)
        offsets = motion_nodes._step_offsets(steps)
        values = {
            "minimax_keyframes": [
                {"resolved_frame_index": 0, MC_KEY: offset, "latent": tail[:, :, index:index + 1]}
                for index, offset in enumerate(offsets)
            ],
            "minimax_frame_count": motion_nodes._pixel_frames(int(target_video.shape[2])),
        }
        if use_audio_context:
            if not payload_patch_applied():
                raise RuntimeError("H3 Motion Context payload 补丁未启用，无法安全使用音频上下文")
            audio_tail, audio_steps, overhang = motion_nodes._audio_tail_from_latent(context_latent, run)
            values["minimax_refs"] = [{
                "kind": "audio", "ref_audio_t": audio_steps,
                "audio_latent": audio_tail.to(device=target_video.device, dtype=target_video.dtype),
                # Match Motion Context's placement exactly: H3's final audio
                # latent can extend a fraction of a step beyond the last video frame.
                MC_AUDIO_KEY: float(run) + float(overhang) / (5.0 / 3.0),
            }]
        return _mark_motion_context(node_helpers.conditioning_set_values(conditioning, values)), run

    def apply(self, conditioning, vae, latent, context_frames, use_video_context, use_audio_context, context_length, context_latent=None, use_video_latent=True):
        if not use_video_context:
            if bool(use_audio_context) and context_latent is not None:
                try:
                    return self._direct_latent_context(conditioning, latent, context_latent, context_length,
                                                       True, use_video_context=False)
                except (ValueError, RuntimeError) as exc:
                    LOG.info("H3 Auto Director: 音频上下文直取不可用，跳过音频上下文：%s", exc)
            return (conditioning, 0)
        if bool(use_video_latent) and context_latent is not None:
            try:
                return self._direct_latent_context(conditioning, latent, context_latent, context_length, use_audio_context)
            except ValueError as exc:
                LOG.info("H3 Auto Director: 视频 latent 直取不可用，回退 VAE 上下文编码：%s", exc)
        cls = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3MotionContext")
        if cls is None:
            raise RuntimeError("Install ComfyUI-H3-Motion-Context before using H3 Auto Director")
        inner = cls()
        result = inner.apply(conditioning, vae, latent, context_frames, context_length,
                             "video", "head", "disabled", context_length,
                             "timeline", context_latent if use_audio_context else None, None, None)
        return (_mark_motion_context(result[0]), result[1])


def _write_wav(path: Path, audio):
    waveform = audio.get("waveform")
    if waveform is None:
        return None
    samples = waveform[0].detach().cpu().clamp(-1, 1).movedim(0, 1).numpy()
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(samples.shape[1])
        out.setsampwidth(2)
        out.setframerate(int(audio.get("sample_rate", 32000)))
        out.writeframes(pcm)
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
            "audio": ("AUDIO",),
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

    def save(self, plan, segment_index, latent, images, fps, output_root="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量", color_correction="关闭", scene_cut_protection=True, scene_cut_threshold=0.18, correction_strength=0.75, residual_strength=0.2, audio=None):
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
        images_to_save = images
        segment_number = int(segment_index)
        correction_mode = str(color_correction)
        if correction_mode in {"匹配首段", "匹配上段"} and segment_number <= 1:
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
        # Context is intentionally written from the raw decoded frames. The
        # next segment must never inherit display-only color correction.
        _write_segment_video(context_path, images, audio, fps, video_format, video_codec, encoder_device, quality)
        if images_to_save is images:
            shutil.copy2(context_path, video_path)
        else:
            _write_segment_video(video_path, images_to_save, audio, fps, video_format, video_codec, encoder_device, quality)
        st_save({"video": parts[0].detach().cpu().contiguous(), "audio": parts[1].detach().cpu().contiguous()}, str(latent_path), metadata={"format": "h3_auto_director_av_v1", "segment_index": str(int(segment_index))})
        return (str(video_path), str(latent_path))


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
    def _assemble(plan, output_name="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量"):
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
        for index in range(1, len(plan.get("segments", [])) + 1):
            clip, _ = _paths(plan, index, output_name)
            if not clip.is_file():
                raise FileNotFoundError("Cannot assemble; missing segment video: %s" % clip)
            lines.append("file '%s'" % str(clip).replace("'", "'\\''"))
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp = final_path.with_name(final_path.stem + ".final_tmp" + final_path.suffix)
        _encode_concat_with_fallback(ffmpeg, list_path, tmp, fmt, codec, encoder_device, quality)
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

    def advance(self, plan, segment_index, saved_video, segment_node_id, output_root="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量", cleanup_after_final=True, prompt=None, client_id=None):
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
        state.setdefault("segments", {})[str(int(segment_index))] = {"status": "completed", saved_key: saved_video}
        state["last_completed"] = int(segment_index)
        _atomic_json(state_path, state)
        if int(segment_index) >= total or not bool(plan.get("auto_run", True)):
            final_path = ""
            if int(segment_index) >= total:
                if str(plan.get("mode", "")) == "tts" and not bool(plan.get("concat_final_audio", True)):
                    final_path = ""
                    state["final_audio"] = ""
                else:
                    final_path = self._assemble(runtime, output_root, video_format, video_codec, encoder_device, quality)
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
    def _assemble(plan, output_name="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量"):
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
    """Select the v0.31 AV audio sampler or the v0.30 legacy audio sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "sampling_mode": ([NATIVE_MODE, LEGACY_MODE], {"default": NATIVE_MODE}),
            "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("模型",)
    FUNCTION = "apply"
    CATEGORY = "H3 自动导演/音频采样"

    def apply(self, model, sampling_mode, shift_video, shift_audio):
        return (apply_h3_sampling(model, sampling_mode, float(shift_video), float(shift_audio)),)


NODE_CLASS_MAPPINGS = {
    "H3AutoDirectorPlan": H3AutoDirectorPlan,
    "H3AutoDirectorTTSPlan": H3AutoDirectorTTSPlan,
    "H3AutoDirectorVideoTransferPlan": H3AutoDirectorVideoTransferPlan,
    "H3AutoDirectorHybridModelLoader": H3AutoDirectorHybridModelLoader,
    "H3AutoDirectorTransferModelLoader": H3AutoDirectorTransferModelLoader,
    "H3AutoDirectorSegment": H3AutoDirectorSegment,
    "H3AutoDirectorReferenceResolver": H3AutoDirectorReferenceResolver,
    "H3AutoDirectorCachedReferenceToVideo": H3AutoDirectorCachedReferenceToVideo,
    "H3AutoDirectorResolution": H3AutoDirectorResolution,
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
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AutoDirectorPlan": "H3 自动导演｜项目计划",
    "H3AutoDirectorTTSPlan": "H3 自动导演｜TTS 项目计划",
    "H3AutoDirectorVideoTransferPlan": "H3 自动导演｜动作迁移项目计划",
    "H3AutoDirectorHybridModelLoader": "H3 自动导演｜多模态参考模型加载",
    "H3AutoDirectorTransferModelLoader": "H3 自动导演｜动作迁移模型加载",
    "H3AutoDirectorSegment": "H3 自动导演｜片段设置",
    "H3AutoDirectorReferenceResolver": "H3 自动导演｜多模态素材解析",
    "H3AutoDirectorCachedReferenceToVideo": "MiniMax H3 多模态参考生成｜提示词缓存",
    "H3AutoDirectorResolution": "H3 自动导演｜双采样分辨率",
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
}
