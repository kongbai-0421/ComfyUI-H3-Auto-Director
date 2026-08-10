"""Native file selection and import for H3 reference assets."""

from pathlib import Path
import ctypes
import os
import shutil

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


def _initial_directory(value, destination):
    requested = str(value or "").strip().strip('"')
    path = Path(requested) if requested else destination
    if not path.is_absolute():
        path = Path(folder_paths.get_input_directory()) / path
    path = path.resolve()
    return path if path.is_dir() else destination


def _select_paths(initial_dir, config):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("ComfyUI Python 缺少 tkinter，无法打开系统文件选择器") from exc

    if os.name == "nt":
        # Make the native Windows dialog follow the monitor's actual scale.
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)  # PER_MONITOR_AWARE_V2
        except (AttributeError, OSError):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
            except (AttributeError, OSError):
                pass

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    patterns = " ".join(f"*{extension}" for extension in sorted(config["extensions"]))
    try:
        return filedialog.askopenfilenames(
            title=f"选择{config['label']}参考素材（可多选）",
            initialdir=str(initial_dir),
            filetypes=[(config["label"], patterns), ("所有文件", "*.*")],
        )
    finally:
        root.destroy()


def _unique_destination(directory, source):
    candidate = directory / source.name
    if not candidate.exists():
        return candidate
    for index in range(1, 10000):
        candidate = directory / f"{source.stem}_{index}{source.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("目标素材目录中的同名文件过多，无法生成唯一文件名")


def select_and_import_files(file_type, initial_dir=""):
    config = _TYPE_CONFIG.get(str(file_type))
    if config is None:
        raise ValueError("不支持的素材类型")
    destination = (Path(folder_paths.get_input_directory()) / "h3_refs" / config["folder"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected = _select_paths(_initial_directory(initial_dir, destination), config)
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
        })
    return results
