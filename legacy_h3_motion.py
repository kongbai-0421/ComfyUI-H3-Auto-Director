"""Compatibility patches for pre-native-guide MiniMax H3 cores.

ComfyUI builds before the native ``MiniMaxH3AddGuide`` node only accept
first/last-frame keyframes.  Auto Director carries context at arbitrary
positions, so it owns the small layout/payload compatibility layer rather
than requiring a separate custom node.
"""

from __future__ import annotations

import inspect
import logging

import torch

import comfy.ldm.minimax.model as mm
import comfy.model_base as model_base

LOG = logging.getLogger("h3_auto_director")
MC_KEY = "h3_auto_director_legacy_frame_index"
MC_AUDIO_KEY = "h3_auto_director_legacy_audio_end_frame"

_layout_applied = False
_payload_applied = False
_original_layout_init = None
_original_extra_conds = None


def _ref_span(refs):
    cursor = 0.0
    for ref in refs or ():
        kind = ref.get("kind")
        if kind == "image":
            cursor += 1.0
        elif kind == "audio":
            cursor += float(ref.get("ref_audio_t", 0))
        elif kind in {"video", "video_audio"}:
            cursor += max(float(ref.get("ref_audio_t", 0)),
                          sum(mm._video_t_spans(int(ref.get("latent_t", 0)))))
    return cursor


def _frame_time(text_len, latent_t, frame_count, frame_index):
    if frame_index == 0:
        return float(text_len)
    if frame_count is not None and frame_index == frame_count - 1:
        return float(text_len) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
    return float(text_len) + mm.FRAME_RESCALE * float(frame_index)


def _patch_layout(self, text_len, latent_t, _latent_h, _latent_w, _audio_t,
                  keyframes=None, refs=None, frame_count=None):
    marked = bool(keyframes) and any(item.get(MC_KEY) is not None for item in keyframes)
    marked_audio = bool(refs) and any(item.get(MC_AUDIO_KEY) is not None for item in refs)
    _original_layout_init(self, text_len, latent_t, _latent_h, _latent_w, _audio_t,
                          keyframes=keyframes, refs=refs, frame_count=frame_count)
    if marked:
        spans = [(start, end) for start, end, kind in self.segments if kind == "cond"]
        if len(spans) != len(keyframes):
            raise RuntimeError("H3 Auto Director legacy context layout changed; cannot place video context")
        offset = _ref_span(refs)
        for (start, end), item in zip(spans, keyframes):
            frame = item.get(MC_KEY)
            if frame is not None:
                self.position_ids[start:end, 0] = _frame_time(text_len, latent_t, frame_count, frame) + offset
    if marked_audio:
        audio_refs = [item for item in refs if item.get(MC_AUDIO_KEY) is not None]
        if len(audio_refs) != 1 or len(refs) != 1:
            raise RuntimeError("H3 Auto Director legacy audio context requires one dedicated audio reference")
        ref = audio_refs[0]
        steps = int(ref.get("ref_audio_t", 0))
        if steps <= 0:
            return
        # Legacy ref audio rows begin at text_len; shift that block to the
        # target timeline while keeping its original per-step spacing.
        time = self.position_ids[:, 0]
        selected = (time >= float(text_len) - 1e-4) & (time < float(text_len) + steps - 1e-4)
        for start, end, kind in self.segments:
            if kind == "cond":
                selected[start:end] = False
        target_origin = float(text_len) + _ref_span(refs)
        shift = target_origin + mm.FRAME_RESCALE * float(ref[MC_AUDIO_KEY]) - steps - float(text_len)
        self.position_ids[selected, 0] = time[selected] + shift


def _patch_payload(self, **kwargs):
    output = _original_extra_conds(self, **kwargs)
    keyframes, refs = kwargs.get("minimax_keyframes"), kwargs.get("minimax_refs")
    if not keyframes or not refs:
        return output
    cond = output.get("minimax_payload")
    payload = getattr(cond, "cond", None)
    if not isinstance(payload, dict):
        LOG.warning("H3 Auto Director: 旧版 H3 payload 不可访问，无法合并视频与音频上下文")
        return output
    payload["cond_video_latents"] = ([item["latent"] for item in keyframes if item.get("latent") is not None]
                                      + [item["latent"] for item in refs if item.get("latent") is not None])
    payload["cond_audio_latents"] = [item["audio_latent"] for item in refs
                                      if item.get("audio_latent") is not None]
    frame_count = kwargs.get("minimax_frame_count")
    if frame_count is not None:
        payload["frame_count"] = frame_count
    return output


def ensure_legacy_h3_motion_context():
    """Install compatibility only when this core has no native arbitrary guide API."""
    global _layout_applied, _payload_applied, _original_layout_init, _original_extra_conds
    try:
        layout_parameters = inspect.signature(mm.PackedLayout.__init__).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    if "frame_count" not in layout_parameters:
        return False
    if not _layout_applied:
        _original_layout_init = mm.PackedLayout.__init__
        mm.PackedLayout.__init__ = _patch_layout
        _layout_applied = True
        LOG.info("H3 Auto Director: 已启用内置旧版 H3 视频上下文布局兼容层")
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is not None and hasattr(cls, "extra_conds") and not _payload_applied:
        _original_extra_conds = cls.extra_conds
        cls.extra_conds = _patch_payload
        _payload_applied = True
        LOG.info("H3 Auto Director: 已启用内置旧版 H3 视频/音频上下文 payload 兼容层")
    return _layout_applied and _payload_applied
