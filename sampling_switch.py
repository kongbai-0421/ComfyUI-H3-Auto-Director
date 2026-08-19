"""MiniMax H3 sampling compatibility implementations.

The legacy branch is based on starsFriday's ComfyUI-MiniMax-H3-LegacySampling
node, but is kept self-contained so users do not need to install that plugin.
"""

from __future__ import annotations

import comfy.model_sampling
import comfy.patcher_extension
import logging
import torch
import torch.nn.functional as F

try:
    import comfy.ldm.minimax.model as _minimax_model
except ImportError:  # pragma: no cover - only older ComfyUI builds
    _minimax_model = None


PATCH_KEY = "h3_auto_director_legacy_audio_sampling"
NATIVE_LAYOUT_PATCH_KEY = "h3_auto_director_native_layout_refresh"
_LAYOUT_REFRESH_MARKER = "h3_auto_director_legacy_layout_refreshed"
_LOG = logging.getLogger("h3_auto_director")
_NATIVE_REFRESH_LOGGED = False
NATIVE_MODE = "ComfyUI v0.31.0版本方法"
LEGACY_MODE = "ComfyUI v0.30.0版本方法"
_MODE_ALIASES = {
    "标准音频采样": NATIVE_MODE,
    "兼容音频采样": LEGACY_MODE,
    "当前版（ComfyUI v0.31.0，AV 音频调度）": NATIVE_MODE,
    "旧版（ComfyUI v0.30.0，Legacy 音频调度）": LEGACY_MODE,
    "当前版（AV 音频调度）": NATIVE_MODE,
    "旧版（Legacy 音频调度）": LEGACY_MODE,
    "当前版": NATIVE_MODE,
    "旧版": LEGACY_MODE,
}
_CONST = getattr(comfy.model_sampling, "CONST", object)


def _is_h3_model(diffusion_model) -> bool:
    model_type = getattr(_minimax_model, "MiniMaxH3Model", None)
    if model_type is not None and isinstance(diffusion_model, model_type):
        return True
    return diffusion_model.__class__.__name__ == "MiniMaxH3Model"


def time_shift_slope(sigma, from_shift, to_shift):
    """Derivative conversion used by the pre-AV H3 audio schedule."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (
        from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    )


class MiniMaxH3LegacyModelSampling(
    comfy.model_sampling.ModelSamplingDiscreteFlow,
    _CONST,
):
    @property
    def audio_scale(self):
        return 1.0


def _resize_keyframe_latent(latent, target_h, target_w):
    """Adapt native H3 guide keyframes to the active target video grid."""
    if not torch.is_tensor(latent) or latent.ndim != 5:
        return latent
    target_h, target_w = int(target_h), int(target_w)
    if tuple(int(v) for v in latent.shape[-2:]) == (target_h, target_w):
        return latent
    b, c, t, h, w = latent.shape
    if min(target_h, target_w, h, w) <= 0:
        return latent
    flat = latent.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    flat = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return flat.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).to(
        device=latent.device, dtype=latent.dtype
    )


def _target_video_from_x(x):
    try:
        return x[0]
    except (IndexError, KeyError, TypeError):
        return None


def _refresh_legacy_h3_payload(kwargs, transformer_options, target_video=None):
    """Drop cached H3 layout data before running the legacy sampler.

    H3 conditioning can be cached between segments, while the packed target
    audio length changes with each segment.  The v0.31 core normally detects
    this through ``PackedLayout.signature``; the v0.30 sampling wrapper can
    receive a layout produced from the previous segment before that check.
    Rebuild the layout from the actual ``x`` shape on every legacy forward and
    rebuild the condition-latent lists from their source blocks as well.
    """
    payload = kwargs.get("minimax_payload")
    if not isinstance(payload, dict):
        return
    if "layout" not in payload and "keyframes" not in payload and "refs" not in payload:
        return
    refreshed = dict(payload)
    refreshed.pop("layout", None)
    keyframes = refreshed.get("keyframes") or []
    refs = refreshed.get("refs") or []
    if torch.is_tensor(target_video) and target_video.ndim == 5:
        target_h, target_w = int(target_video.shape[-2]), int(target_video.shape[-1])
        adapted = []
        for keyframe in keyframes:
            if not isinstance(keyframe, dict):
                adapted.append(keyframe)
                continue
            item = dict(keyframe)
            item["latent"] = _resize_keyframe_latent(item.get("latent"), target_h, target_w)
            adapted.append(item)
        keyframes = adapted
        refreshed["keyframes"] = keyframes
    if keyframes or refs:
        refreshed["cond_video_latents"] = [item["latent"] for item in keyframes
                                            if isinstance(item, dict) and item.get("latent") is not None]
        refreshed["cond_video_latents"] += [item["latent"] for item in refs
                                             if isinstance(item, dict) and item.get("latent") is not None]
        refreshed["cond_audio_latents"] = [item["audio_latent"] for item in keyframes
                                            if isinstance(item, dict) and item.get("audio_latent") is not None]
        refreshed["cond_audio_latents"] += [item["audio_latent"] for item in refs
                                             if isinstance(item, dict) and item.get("audio_latent") is not None]
    kwargs["minimax_payload"] = refreshed
    if not transformer_options.get(_LAYOUT_REFRESH_MARKER):
        transformer_options[_LAYOUT_REFRESH_MARKER] = True


def _remove_legacy_motion_audio(payload):
    """Remove only Auto Director's audio continuation blocks for a retry."""
    if not isinstance(payload, dict):
        return None
    keyframes = payload.get("keyframes") or []
    refs = payload.get("refs") or []
    changed = False
    new_keyframes = []
    for item in keyframes:
        if not isinstance(item, dict):
            new_keyframes.append(item)
            continue
        clone = dict(item)
        if clone.get("_h3_auto_director_native_context") and clone.get("audio_latent") is not None:
            clone.pop("audio_latent", None)
            changed = True
        new_keyframes.append(clone)
    new_refs = []
    for item in refs:
        if not isinstance(item, dict):
            new_refs.append(item)
            continue
        if ("h3_auto_director_legacy_audio_end_frame" in item
                or "motion_context_audio_end_frame" in item):
            changed = True
            continue
        new_refs.append(item)
    if not changed:
        return None
    retry = dict(payload)
    retry.pop("layout", None)
    retry["keyframes"] = new_keyframes
    retry["refs"] = new_refs
    retry["cond_video_latents"] = [item["latent"] for item in new_keyframes
                                    if isinstance(item, dict) and item.get("latent") is not None]
    retry["cond_video_latents"] += [item["latent"] for item in new_refs
                                     if isinstance(item, dict) and item.get("latent") is not None]
    retry["cond_audio_latents"] = [item["audio_latent"] for item in new_keyframes
                                    if isinstance(item, dict) and item.get("audio_latent") is not None]
    retry["cond_audio_latents"] += [item["audio_latent"] for item in new_refs
                                     if isinstance(item, dict) and item.get("audio_latent") is not None]
    return retry


def _is_audio_layout_error(exc):
    message = str(exc)
    return ("expanded size" in message and "audio_embed" in message) or (
        "expanded size" in message and "5376" in message
    )


def legacy_audio_sampling_wrapper(executor, x, timestep, context, transformer_options, **kwargs):
    _refresh_legacy_h3_payload(kwargs, transformer_options, _target_video_from_x(x))
    try:
        output = executor(x, timestep, context, transformer_options, **kwargs)
    except RuntimeError as exc:
        payload = kwargs.get("minimax_payload")
        retry_payload = _remove_legacy_motion_audio(payload) if _is_audio_layout_error(exc) else None
        if retry_payload is None:
            raise
        _LOG.warning(
            "H3 Auto Director: v0.30 音频上下文布局不兼容，已移除自动导演音频上下文并保留视频上下文后重试"
        )
        kwargs["minimax_payload"] = retry_payload
        output = executor(x, timestep, context, transformer_options, **kwargs)
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise RuntimeError("MiniMax H3 legacy sampling expected separate video/audio model outputs")
    diffusion_model = executor.class_obj
    shift_video = float(
        transformer_options.get("minimax_h3_sigma_shift_video", diffusion_model.sigma_shift_video)
    )
    shift_audio = float(
        transformer_options.get("minimax_h3_sigma_shift_audio", diffusion_model.sigma_shift_audio)
    )
    sigma_video = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    slope_audio = time_shift_slope(sigma_video, shift_video, shift_audio).to(output[1].dtype)
    return [output[0], output[1] * slope_audio]


def native_layout_refresh_wrapper(executor, x, timestep, context, transformer_options, **kwargs):
    """Refresh cached H3 condition layout while keeping native AV behavior."""
    global _NATIVE_REFRESH_LOGGED
    if not _NATIVE_REFRESH_LOGGED:
        _LOG.info("H3 Auto Director: v0.31 原生音频采样已启用跨片段布局刷新")
        _NATIVE_REFRESH_LOGGED = True
    _refresh_legacy_h3_payload(kwargs, transformer_options, _target_video_from_x(x))
    return executor(x, timestep, context, transformer_options, **kwargs)


def _new_model_sampling(model, sampling_type, shift_video, shift_audio):
    """Build the requested sampling object while preserving model noise scale."""
    model_sampling_type = (
        MiniMaxH3LegacyModelSampling
        if sampling_type == "ModelSamplingDiscreteFlow"
        else getattr(comfy.model_sampling, sampling_type, None)
    )
    if model_sampling_type is None:
        raise RuntimeError(
            "当前 ComfyUI 缺少 %s，无法使用%s；请升级 ComfyUI 或切换到可用模式。"
            % (sampling_type, "当前版" if sampling_type == "ModelSamplingAV" else "旧版")
        )
    class_sampling = type(
        "H3AutoDirectorModelSampling",
        (model_sampling_type, _CONST),
        {},
    )
    sampling = class_sampling(model.model.model_config)
    original = model.get_model_object("model_sampling")
    if sampling_type == "ModelSamplingAV":
        sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
    else:
        sampling.set_parameters(shift=shift_video)
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    return sampling


def apply_h3_sampling(model, mode, shift_video, shift_audio):
    """Clone and patch an H3 model for either native or legacy sampling."""
    mode = _MODE_ALIASES.get(str(mode), str(mode))
    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    if not _is_h3_model(diffusion_model):
        raise ValueError("MiniMax H3 音频采样切换仅支持 MiniMax H3 模型")

    if mode == LEGACY_MODE:
        if not hasattr(comfy.patcher_extension, "WrappersMP"):
            raise RuntimeError("当前 ComfyUI 不支持模型包装器，无法使用兼容音频采样")
        sampling = _new_model_sampling(patched, "ModelSamplingDiscreteFlow", shift_video, shift_audio)
        patched.add_object_patch("model_sampling", sampling)
        transformer_options = patched.model_options["transformer_options"] = patched.model_options.get(
            "transformer_options", {}
        ).copy()
        transformer_options["minimax_h3_sigma_shift_video"] = float(shift_video)
        transformer_options["minimax_h3_sigma_shift_audio"] = float(shift_audio)
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, PATCH_KEY)
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, NATIVE_LAYOUT_PATCH_KEY)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            PATCH_KEY,
            legacy_audio_sampling_wrapper,
        )
        _LOG.info("H3 Auto Director: 音频采样=%s，视频偏移=%.3f，音频偏移=%.3f",
                  LEGACY_MODE, float(shift_video), float(shift_audio))
        return patched

    if mode != NATIVE_MODE:
        raise ValueError("未知 H3 调度器模式: %s" % mode)
    sampling = _new_model_sampling(patched, "ModelSamplingAV", shift_video, shift_audio)
    patched.add_object_patch("model_sampling", sampling)
    transformer_options = patched.model_options["transformer_options"] = patched.model_options.get(
        "transformer_options", {}
    ).copy()
    transformer_options["minimax_h3_sigma_shift_video"] = float(shift_video)
    transformer_options["minimax_h3_sigma_shift_audio"] = float(shift_audio)
    if hasattr(comfy.patcher_extension, "WrappersMP"):
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, PATCH_KEY)
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, NATIVE_LAYOUT_PATCH_KEY)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            NATIVE_LAYOUT_PATCH_KEY,
            native_layout_refresh_wrapper,
        )
    _LOG.info("H3 Auto Director: 音频采样=%s，视频偏移=%.3f，音频偏移=%.3f",
              NATIVE_MODE, float(shift_video), float(shift_audio))
    return patched
