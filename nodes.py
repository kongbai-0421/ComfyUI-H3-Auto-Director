"""Project runner primitives for MiniMax H3 in ComfyUI.

The controller queues the next copy of the current workflow after a segment
has been saved. It deliberately uses numbered project slots, so a rerun never
silently consumes the newest rejected cache.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
import json
import logging
import os
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

try:
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo as _H3ReferenceToVideo
except ImportError:
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


def _paths(plan, index: int, output_name="", video_format="mp4", for_write=False):
    base = Path(plan["project_dir"])
    clips = base / "clips"
    cache = base / "cache"
    if for_write:
        stem = _output_filename(output_name)
        ext = "." + str(video_format or "mp4").lower().lstrip(".")
        if ext not in VIDEO_EXTENSIONS:
            ext = ".mp4"
        return clips / ("%s_%05d%s" % (stem, index, ext)), cache / ("%s_%05d.safetensors" % (stem, index))
    video = _indexed_file(clips, index, VIDEO_EXTENSIONS, output_name)
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
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            tail.append(frame.to_ndarray(format="rgb24"))
    if not tail:
        raise ValueError("Context video has no frames: %s" % path)
    return torch.from_numpy(__import__("numpy").stack(list(tail))).float() / 255.0


def _load_av_latent(path: Path):
    if st_load is None:
        raise RuntimeError("safetensors is required for H3 AV latent caches")
    values = st_load(str(path), device="cpu")
    if "video" not in values or "audio" not in values:
        raise ValueError("Not an H3 AV latent cache: %s" % path)
    return {"samples": [values["video"], values["audio"]]}


def _validate_reference_limits(refs, label="参考素材"):
    """Validate H3 per-segment reference limits before any files are loaded."""
    if not isinstance(refs, list):
        raise ValueError(f"{label}必须是 JSON 列表")
    valid = [ref for ref in refs if isinstance(ref, dict) and (ref.get("path") or ref.get("name"))]
    if len(valid) > MAX_REFERENCE_TOTAL:
        raise ValueError(f"{label}总数最多 {MAX_REFERENCE_TOTAL} 个")
    counts = {"image": 0, "video": 0, "audio": 0}
    for ref in valid:
        kind = str(ref.get("type", "image")).lower()
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
            "continuation_mode": ("BOOLEAN", {"default": True, "tooltip": "开启后从 segment_index 对应的上一段读取视频和 AV latent 上下文"}),
            "cache_prompt_embeddings": ("BOOLEAN", {"default": False, "tooltip": "首次执行时一次性编码并缓存全部片段的多模态提示词向量"}),
            "output_root": ("STRING", {"default": "h3_projects", "tooltip": "项目文件夹名称；新路径为 output/h3_project/<此名称>"}),
        }, "optional": {
            "global_assets_json": ("STRING", {"default": "[]", "multiline": True}),
        }, "hidden": {"project_dir": "STRING"}}

    RETURN_TYPES = ("H3_AUTO_PLAN",)
    RETURN_NAMES = ("项目计划",)
    FUNCTION = "create"
    CATEGORY = "H3 自动导演"

    def create(self, project_id, segments_json, duration, global_reference_set, auto_run, continuation_mode=True, cache_prompt_embeddings=False, output_root="h3_projects", global_assets_json="[]", project_dir=""):
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
            row["continue_video"] = bool(row.get("continue_video", bool(continuation_mode) and len(normalized) > 0))
            row["references"] = list(row.get("references", []))
            _validate_reference_limits(row["references"], f"第 {len(normalized) + 1} 段参考素材")
            normalized.append(row)
        # The UI keeps this field in sync for compatibility, but the global
        # policy is deliberately defined by the first segment itself. This
        # prevents stale hidden JSON in an older workflow from overriding it.
        if global_reference_set:
            assets = list(normalized[0].get("references", []))
        _validate_reference_limits(assets, "统一参考素材")
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
        (project_dir / "clips").mkdir(exist_ok=True)
        (project_dir / "final").mkdir(exist_ok=True)
        _atomic_json(project_dir / "json" / "project.json", {k: v for k, v in plan.items() if k != "project_dir"})
        state = _load_json(_state_path(plan), {"version": 2, "segments": {}})
        state.setdefault("segments", {})
        _atomic_json(project_dir / "json" / "state.json", state)
        return (plan,)


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
        use_video = bool(plan.get("continuation_mode", True)) and bool(seg.get("continue_video", context_index > 0)) and context_index > 0
        restart = bool(seg.get("audio_restart", False))
        use_audio = use_video and not restart
        target = round(float(seg["duration"]) * FPS)
        physical = _align_frames(target + (int(context_length) if use_video else 0))
        refs = plan.get("global_assets", []) if plan.get("global_reference_set", True) else []
        if not plan.get("global_reference_set", True):
            refs = seg.get("references", [])
        return (str(seg.get("prompt", "")), physical, use_video, use_audio,
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


def _resolve_reference_groups(refs):
    """Load references while preserving video/audio pairing indexes."""
    _validate_reference_limits(refs, "每段参考素材")
    images, videos, video_audios, standalone_audios = [], [], [], []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        kind = str(ref.get("type", "image")).lower()
        name = ref.get("path") or ref.get("name")
        if not name:
            continue
        if kind == "image":
            images.append(_load_reference_image(name))
        elif kind == "video":
            frames, soundtrack = _load_reference_video(name)
            videos.append(frames)
            video_audios.append(soundtrack)
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
    if plan.get("global_reference_set", True):
        return list(plan.get("global_assets", []))
    return list(_segment(plan, generation_index).get("references", []))


def _cache_frame_count(plan, generation_index, context_length):
    seg = _segment(plan, generation_index)
    use_video = bool(plan.get("continuation_mode", True)) and bool(seg.get("continue_video", generation_index > 1)) and generation_index > 1
    target = round(float(seg["duration"]) * FPS)
    return _align_frames(target + (int(context_length) if use_video else 0))


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
    def _encode_one(clip, vae, audio_vae, prompt, width, height, length, ref_image_size, refs):
        if _H3ReferenceToVideo is None:
            raise RuntimeError("当前 ComfyUI 未提供 MiniMaxH3ReferenceToVideo 核心节点")
        ref_groups = _resolve_reference_groups(refs)
        result = _H3ReferenceToVideo.execute(
            clip, vae, audio_vae, prompt, int(width), int(height), int(length), str(ref_image_size),
            ref_images=ref_groups[0], ref_videos=ref_groups[1],
            ref_video_audios=ref_groups[2], ref_audios=ref_groups[3])
        return result[0], result[1]

    @classmethod
    def _build_cache(cls, plan, clip, vae, audio_vae, width, height, ref_image_size, context_length):
        cache = {}
        for generation_index in range(1, len(plan.get("segments", [])) + 1):
            seg = _segment(plan, generation_index)
            length = _cache_frame_count(plan, generation_index, context_length)
            cache[generation_index] = cls._encode_one(
                clip, vae, audio_vae, str(seg.get("prompt", "")), width, height, length,
                ref_image_size, _cache_segment_references(plan, generation_index))
        return cache

    @classmethod
    def encode(cls, plan, clip, vae, audio_vae, prompt, width, height, length,
               ref_image_size="match", context_length=FRAME_CONTEXT_DEFAULT, segment_index=0,
               references_json=None):
        generation_index = int(segment_index) + 1
        refs = None
        if references_json is not None and str(references_json).strip():
            try:
                refs = json.loads(references_json)
            except json.JSONDecodeError as exc:
                raise ValueError("参考素材 JSON 无效: %s" % exc) from exc
            _validate_reference_limits(refs, "当前片段参考素材")
        if not bool(plan.get("cache_prompt_embeddings", False)):
            return cls._encode_one(clip, vae, audio_vae, prompt, width, height, length,
                                   ref_image_size, _cache_segment_references(plan, generation_index) if refs is None else refs)
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
            raise ValueError("segment_index %d 对应的下一段不存在" % int(segment_index))
        return cache[generation_index]


class H3AutoDirectorContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("H3_AUTO_PLAN",), "segment_index": ("INT", {"default": 0, "min": 0})}}

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("上下文画面", "上下文潜变量")
    FUNCTION = "load"
    CATEGORY = "H3 自动导演"

    def load(self, plan, segment_index):
        if int(segment_index) <= 0 or not bool(plan.get("continuation_mode", True)):
            return (torch.zeros((1, 1, 1, 3), dtype=torch.float32), {"samples": [torch.zeros((1, 24, 2, 1, 1)), torch.zeros((1, 32, 2, 1))]})
        video_path, latent_path = _paths(plan, int(segment_index))
        if not video_path.exists() or not latent_path.exists():
            raise FileNotFoundError("Missing context cache for segment %d: %s / %s" % (int(segment_index), video_path, latent_path))
        return (_load_context_video(video_path), _load_av_latent(latent_path))


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
        }, "optional": {"context_latent": ("LATENT",)}}

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("条件", "裁剪帧数")
    FUNCTION = "apply"
    CATEGORY = "H3 自动导演"

    def apply(self, conditioning, vae, latent, context_frames, use_video_context, use_audio_context, context_length, context_latent=None):
        if not use_video_context:
            return (conditioning, 0)
        cls = nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3MotionContext")
        if cls is None:
            raise RuntimeError("Install ComfyUI-H3-Motion-Context before using H3 Auto Director")
        inner = cls()
        return inner.apply(conditioning, vae, latent, context_frames, context_length,
                           "video", "head", "disabled", context_length,
                           "timeline", context_latent if use_audio_context else None, None, None)


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
        }, "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("已保存视频", "已保存潜变量")
    FUNCTION = "save"
    CATEGORY = "H3 自动导演"
    OUTPUT_NODE = True

    def save(self, plan, segment_index, latent, images, fps, output_root="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量", audio=None):
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = FPS
        if fps <= 0:
            fps = FPS
        plan = _runtime_plan(plan)
        video_path, latent_path = _paths(plan, int(segment_index), output_root, video_format, for_write=True)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        if st_save is None:
            raise RuntimeError("safetensors is required to save H3 AV context")
        parts = list(latent["samples"].unbind()) if hasattr(latent["samples"], "unbind") else list(latent["samples"])
        if len(parts) < 2:
            raise ValueError("Sampler output must be an H3 video/audio latent pair")
        tmp_video = video_path.with_name(video_path.stem + ".video_tmp" + video_path.suffix)
        _encode_video_with_fallback(tmp_video, images, fps, video_format, video_codec, encoder_device, quality)
        tmp_wav = video_path.with_suffix(".audio_tmp.wav")
        wav = _write_wav(tmp_wav, audio) if audio is not None else None
        if wav is not None:
            ffmpeg = _find_ffmpeg()
            if not ffmpeg:
                tmp_video.unlink(missing_ok=True)
                tmp_wav.unlink(missing_ok=True)
                raise RuntimeError("未找到 ffmpeg，无法封装 H3 片段音频。请重启 ComfyUI，或设置环境变量 FFMPEG_PATH 指向 ffmpeg.exe。")
            try:
                audio_codec = "libopus" if str(video_format).lower() == "webm" else "aac"
                mux_args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp_video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", audio_codec, "-shortest"]
                if str(video_format).lower() in {"mp4", "mov"}:
                    mux_args.extend(["-movflags", "+faststart"])
                mux_args.append(str(video_path))
                subprocess.run(mux_args, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                tmp_video.unlink(missing_ok=True)
                tmp_wav.unlink(missing_ok=True)
                video_path.unlink(missing_ok=True)
                detail = (exc.stderr or b"").decode(errors="replace")[-1200:]
                raise RuntimeError("ffmpeg 无法封装片段音频：%s" % detail) from exc
            tmp_video.unlink(missing_ok=True)
            tmp_wav.unlink(missing_ok=True)
        else:
            os.replace(tmp_video, video_path)
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
        }, "hidden": {"prompt": "PROMPT", "client_id": "CLIENT_ID"}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("运行状态", "最终视频")
    FUNCTION = "advance"
    CATEGORY = "H3 自动导演"
    OUTPUT_NODE = True

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
        os.replace(tmp, final_path)
        return str(final_path)

    def advance(self, plan, segment_index, saved_video, segment_node_id, output_root="", video_format="mp4", video_codec="h264", encoder_device="CPU", quality="最高质量", prompt=None, client_id=None):
        total = len(plan.get("segments", []))
        runtime = dict(plan)
        saved_path = Path(str(saved_video)).resolve() if saved_video else None
        if saved_path and saved_path.is_file() and _output_root() in saved_path.parents:
            runtime["project_dir"] = str(saved_path.parent.parent)
        project_dir = Path(runtime["project_dir"])
        state_path = _state_path(runtime)
        state = _load_json(state_path, {"version": 1, "segments": {}})
        state.setdefault("segments", {})[str(int(segment_index))] = {"status": "completed", "video": saved_video}
        state["last_completed"] = int(segment_index)
        _atomic_json(state_path, state)
        if int(segment_index) >= total or not bool(plan.get("auto_run", True)):
            final_path = ""
            if int(segment_index) >= total:
                final_path = self._assemble(runtime, output_root, video_format, video_codec, encoder_device, quality)
                state["final_video"] = final_path
                state["status"] = "complete"
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
        for data in next_prompt.values():
            if data.get("class_type") == "H3AutoDirectorPlan":
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
        server.prompt_queue.put((number, prompt_id, next_prompt, extra, output_nodes, {}))
        state["status"] = "queued"
        state["next_segment"] = int(segment_index) + 1
        _atomic_json(state_path, state)
        return ("queued segment %d/%d" % (int(segment_index) + 1, total),)


NODE_CLASS_MAPPINGS = {
    "H3AutoDirectorPlan": H3AutoDirectorPlan,
    "H3AutoDirectorSegment": H3AutoDirectorSegment,
    "H3AutoDirectorReferenceResolver": H3AutoDirectorReferenceResolver,
    "H3AutoDirectorCachedReferenceToVideo": H3AutoDirectorCachedReferenceToVideo,
    "H3AutoDirectorContext": H3AutoDirectorContext,
    "H3AutoDirectorResumeContext": H3AutoDirectorResumeContext,
    "H3AutoDirectorMotionContext": H3AutoDirectorMotionContext,
    "H3AutoDirectorSaveSegment": H3AutoDirectorSaveSegment,
    "H3AutoDirectorController": H3AutoDirectorController,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AutoDirectorPlan": "H3 自动导演｜项目计划",
    "H3AutoDirectorSegment": "H3 自动导演｜片段设置",
    "H3AutoDirectorReferenceResolver": "H3 自动导演｜多模态素材解析",
    "H3AutoDirectorCachedReferenceToVideo": "MiniMax H3 多模态参考生成｜提示词缓存",
    "H3AutoDirectorContext": "H3 自动导演｜上下文读取",
    "H3AutoDirectorResumeContext": "H3 自动导演｜断点续接",
    "H3AutoDirectorMotionContext": "H3 自动导演｜运动上下文",
    "H3AutoDirectorSaveSegment": "H3 自动导演｜保存片段",
    "H3AutoDirectorController": "H3 自动导演｜拼接最终视频",
}
