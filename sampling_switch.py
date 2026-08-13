"""MiniMax H3 sampling compatibility implementations.

The legacy branch is based on starsFriday's ComfyUI-MiniMax-H3-LegacySampling
node, but is kept self-contained so users do not need to install that plugin.
"""

from __future__ import annotations

import comfy.model_sampling
import comfy.patcher_extension

try:
    import comfy.ldm.minimax.model as _minimax_model
except ImportError:  # pragma: no cover - only older ComfyUI builds
    _minimax_model = None


PATCH_KEY = "h3_auto_director_legacy_audio_sampling"
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


def legacy_audio_sampling_wrapper(executor, x, timestep, context, transformer_options, **kwargs):
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
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            PATCH_KEY,
            legacy_audio_sampling_wrapper,
        )
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
    return patched
