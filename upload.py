"""Native file selection and import for H3 reference assets."""

from pathlib import Path
import ctypes
import os
import shutil
import subprocess

import folder_paths


_TYPE_CONFIG = {
    "image": {
        "folder": "images",
        "extensions": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
        "label": "图片",
    },
    "video": {
        "folder": "videos",
        "extensions": {".mp4", ".mov", ".webm", ".mkv", ".avi"},
        "label": "视频",
    },
    "audio": {
        "folder": "audio",
        "extensions": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"},
        "label": "音频",
    },
}


def _find_ffprobe():
    configured = os.environ.get("FFPROBE_PATH", "").strip().strip('"')
    candidates = [configured] if configured else []
    found = shutil.which("ffprobe")
    if found:
        candidates.append(found)
    ffmpeg = os.environ.get("FFMPEG_PATH", "").strip().strip('"')
    if ffmpeg:
        candidates.append(str(Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix)))
    for value in candidates:
        if value and Path(value).is_file():
            return str(Path(value))
    return None


def _find_ffmpeg():
    configured = os.environ.get("FFMPEG_PATH", "").strip().strip('"')
    if configured and Path(configured).is_file():
        return configured
    return shutil.which("ffmpeg")


def _video_has_audio(path):
    """Return whether a local video contains an audio stream."""
    ffprobe = _find_ffprobe()
    if ffprobe:
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, timeout=20,
            )
            if result.returncode == 0:
                return bool(result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return False
    try:
        result = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                                capture_output=True, text=True, timeout=20)
        return "Audio:" in result.stderr
    except (OSError, subprocess.SubprocessError):
        return False


def _resolve_input_file(value):
    clean = str(value or "").strip().strip('"').replace("\\", "/")
    if clean.startswith("input/"):
        clean = clean[6:]
    path = (Path(folder_paths.get_input_directory()) / clean).resolve()
    root = Path(folder_paths.get_input_directory()).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("视频参考素材必须位于 ComfyUI/input 目录内")
    return path


def probe_video_file(path):
    return {"has_audio": _video_has_audio(_resolve_input_file(path))}


def _initial_directory(value, destination):
    requested = str(value or "").strip().strip('"')
    path = Path(requested) if requested else destination
    if not path.is_absolute():
        path = Path(folder_paths.get_input_directory()) / path
    path = path.resolve()
    return path if path.is_dir() else destination


def _enable_windows_dpi_awareness():
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)  # PER_MONITOR_AWARE_V2
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
        except (AttributeError, OSError):
            pass


def _create_hidden_root():
    try:
        import tkinter as tk
    except ImportError as exc:
        raise RuntimeError("ComfyUI Python 缺少 tkinter，无法打开系统文件选择器") from exc
    _enable_windows_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def _select_paths(initial_dir, config):
    from tkinter import filedialog

    root = _create_hidden_root()
    patterns = " ".join(f"*{extension}" for extension in sorted(config["extensions"]))
    try:
        options = {
            "title": f"选择{config['label']}参考素材（可多选）",
            "filetypes": [(config["label"], patterns), ("所有文件", "*.*")],
        }
        if initial_dir is not None:
            options["initialdir"] = str(initial_dir)
        return filedialog.askopenfilenames(**options)
    finally:
        root.destroy()


def select_directory(initial_dir=""):
    from tkinter import filedialog

    destination = Path(folder_paths.get_input_directory()).resolve()
    root = _create_hidden_root()
    try:
        selected = filedialog.askdirectory(title="选择文件选择器默认打开目录", initialdir=str(_initial_directory(initial_dir, destination)))
    finally:
        root.destroy()
    return str(Path(selected).resolve()) if selected else ""


def _unique_destination(directory, source):
    candidate = directory / source.name
    if not candidate.exists():
        return candidate
    for index in range(1, 10000):
        candidate = directory / f"{source.stem}_{index}{source.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("目标素材目录中的同名文件过多，无法生成唯一文件名")


def select_and_import_files(file_type, initial_dir="", use_default_path=True):
    config = _TYPE_CONFIG.get(str(file_type))
    if config is None:
        raise ValueError("不支持的素材类型")
    destination = (Path(folder_paths.get_input_directory()) / "h3_refs" / config["folder"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    start_dir = _initial_directory(initial_dir, destination) if use_default_path else None
    selected = _select_paths(start_dir, config)
    results = []
    for value in selected:
        source = Path(value).resolve()
        if not source.is_file() or source.suffix.lower() not in config["extensions"]:
            continue
        target = source if source.parent == destination else _unique_destination(destination, source)
        if target != source:
            shutil.copy2(source, target)
        results.append({
            "type": file_type,
            "name": target.name,
            "path": f"h3_refs/{config['folder']}/{target.name}",
            "originalName": source.name,
            **({"has_audio": _video_has_audio(target), "video_audio_enabled": True} if file_type == "video" else {}),
        })
    return results
