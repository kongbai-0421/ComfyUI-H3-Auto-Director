import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "H3AutoDirectorPlan";
const TRANSFER_NODE = "H3AutoDirectorVideoTransferPlan";
const TRANSFER_LOADER_NODE = "H3AutoDirectorTransferModelLoader";
const HYBRID_LOADER_NODE = "H3AutoDirectorHybridModelLoader";
const DUAL_STAGE_LOADER_NODE = "H3AutoDirectorDualStageModelLoader";
const TTS_NODE = "H3AutoDirectorTTSPlan";
const SEGMENT_NODE = "H3AutoDirectorSegment";
const REFERENCE_NODE = "H3AutoDirectorReferenceResolver";
const CACHED_REFERENCE_NODE = "H3AutoDirectorCachedReferenceToVideo";
const DUAL_SAMPLING_NODE = "H3AutoDirectorDualSampling";
const AV_DECODE_NODE = "H3AutoDirectorAVDecode";
const CONTEXT_NODE = "H3AutoDirectorContext";
const RESUME_NODE = "H3AutoDirectorResumeContext";
const MOTION_CONTEXT_NODE = "H3AutoDirectorMotionContext";
const MOTION_TRIM_NODE = "MiniMaxH3MotionContextTrim";
const MOTION_SAVE_LATENT_NODE = "MiniMaxH3MotionContextSaveLatent";
const MOTION_LOAD_LATENT_NODE = "MiniMaxH3MotionContextLoadLatent";
const RESOLUTION_NODE = "ResolutionSelector";
const H3_RESOLUTION_NODE = "H3AutoDirectorResolution";
const SAVE_NODE = "H3AutoDirectorSaveSegment";
const CONTROLLER_NODE = "H3AutoDirectorController";
const SAMPLING_SWITCH_NODE = "H3AutoDirectorSamplingSwitch";
const APPLY_AUDIO_SAMPLING_NODE = "H3AutoDirectorApplyAudioSampling";
const CONTROL_PREPROCESS_NODE = "H3AutoDirectorControlPreprocess";
const CONTROL_CONFIG_NODE = "H3AutoDirectorControlConfig";
const CONTROL_EXPORT_NODE = "H3AutoDirectorControlExport";
const CONTROL_CHECK_NODE = "H3AutoDirectorControlBackendCheck";
// Retired video super-resolution nodes are intentionally kept as sentinel
// names so old saved graphs can be recognized without re-enabling their UI.
const VIDEO_LOAD_NODE = "__H3_RETIRED_VIDEO_LOAD__";
const VIDEO_STREAM_NODE = "__H3_RETIRED_VIDEO_STREAM__";
const LOAD_SAVED_AV_LATENT_NODE = "H3AutoDirectorLoadSavedAVLatent";
const DECODE_SAVE_VIDEO_NODE = "H3AutoDirectorDecodeSaveVideo";
const H3_NODE_CLASSES = new Set([NODE, TRANSFER_NODE, TRANSFER_LOADER_NODE, DUAL_STAGE_LOADER_NODE, TTS_NODE, SEGMENT_NODE, REFERENCE_NODE, CACHED_REFERENCE_NODE, DUAL_SAMPLING_NODE, AV_DECODE_NODE, CONTEXT_NODE, RESUME_NODE, MOTION_CONTEXT_NODE, MOTION_TRIM_NODE, MOTION_SAVE_LATENT_NODE, MOTION_LOAD_LATENT_NODE, RESOLUTION_NODE, H3_RESOLUTION_NODE, SAVE_NODE, CONTROLLER_NODE, SAMPLING_SWITCH_NODE, APPLY_AUDIO_SAMPLING_NODE, CONTROL_PREPROCESS_NODE, CONTROL_CONFIG_NODE, CONTROL_EXPORT_NODE, CONTROL_CHECK_NODE, LOAD_SAVED_AV_LATENT_NODE, DECODE_SAVE_VIDEO_NODE, HYBRID_LOADER_NODE]);
const MAX_REFS = { image: 9, video: 3, audio: 3 };
const MAX_TOTAL_REFS = 12;
const DIR_KEY = "h3-auto-director-picker-dirs";
const PICKER_OPTIONS_KEY = "h3-auto-director-picker-options";
const UPLOAD_DIRS = { image: "h3_refs/images", video: "h3_refs/videos", audio: "h3_refs/audio" };
const VIDEO_AUDIO_PROBES = new WeakSet();

function widget(node, name) {
  return (node.widgets || []).find((item) => item.name === name);
}

function syncSerializedWidgets(node) {
  if (!node || !Array.isArray(node.widgets)) return;
  // LiteGraph excludes non-serialized buttons from widgets_values. Rebuild
  // with the same rule so hidden JSON widgets keep their correct positions.
  node.widgets_values = node.widgets
    .filter((item) => item && item.serialize !== false)
    .map((item) => item.value);
}

function isRetiredPort(port, names) {
  if (!port) return false;
  return [port.name, port.label, port.localized_name]
    .some((value) => names.has(String(value || "")));
}

function removeRetiredPorts(node, inputNames = [], outputNames = inputNames) {
  if (!node) return false;
  const inputs = new Set(inputNames);
  const outputs = new Set(outputNames);
  let changed = false;
  // Remove backwards-compatible sockets from the end so LiteGraph indices
  // remain stable while links are detached by removeInput/removeOutput.
  for (let index = (node.inputs || []).length - 1; index >= 0; index -= 1) {
    if (!isRetiredPort(node.inputs[index], inputs)) continue;
    node.removeInput?.(index);
    changed = true;
  }
  for (let index = (node.outputs || []).length - 1; index >= 0; index -= 1) {
    if (!isRetiredPort(node.outputs[index], outputs)) continue;
    node.removeOutput?.(index);
    changed = true;
  }
  if (changed) node.setDirtyCanvas?.(true, true);
  return changed;
}

function cleanDualStageLoaderPorts(node) {
  const names = ["stage1_sigmas", "stage2_sigmas", "一采 Sigmas 调度", "二采 Sigmas 调度",
    "一采 Sigmas", "二采 Sigmas", "一采 Sigma", "二采 Sigma"];
  return removeRetiredPorts(node, names, names);
}

function cleanSamplingSwitchPorts(node) {
  const changed = removeRetiredPorts(node,
    ["scheduler", "steps", "denoise", "调度器", "采样步数", "降噪"],
    ["模型", "MODEL", "视频调度偏移", "音频调度偏移"]);
  let added = false;
  if (!(node.outputs || []).some((output) => String(output?.type || "") === "H3_AUDIO_SAMPLING")) {
    node.addOutput?.("采样调度信息", "H3_AUDIO_SAMPLING");
    added = true;
  }
  if (!(node.outputs || []).some((output) => String(output?.type || "") === "SIGMAS")) {
    node.addOutput?.("SIGMAS", "SIGMAS");
    added = true;
  }
  for (const output of node.outputs || []) {
    if (String(output?.type || "") !== "H3_AUDIO_SAMPLING") continue;
    output.name = "采样调度信息";
    output.label = "采样调度信息";
    output.localized_name = "采样调度信息";
  }
  if (added) {
    node.setDirtyCanvas?.(true, true);
    return true;
  }
  return changed;
}

function readDirectories() {
  try {
    return { image: "h3_refs/images", video: "h3_refs/videos", audio: "h3_refs/audio", ...JSON.parse(localStorage.getItem(DIR_KEY) || "{}") };
  } catch (_) {
    return { image: "h3_refs/images", video: "h3_refs/videos", audio: "h3_refs/audio" };
  }
}

function writeDirectories(dirs) {
  localStorage.setItem(DIR_KEY, JSON.stringify(dirs));
}

function readPickerOptions() {
  try {
    const saved = JSON.parse(localStorage.getItem(PICKER_OPTIONS_KEY) || "{}");
    return { useDefaultPath: saved.useDefaultPath !== false, mode: saved.mode === "browser" ? "browser" : "python" };
  } catch (_) {
    return { useDefaultPath: true, mode: "python" };
  }
}

function writePickerOptions(options) {
  localStorage.setItem(PICKER_OPTIONS_KEY, JSON.stringify(options));
}

function normalizeSegment(value) {
  const seg = value && typeof value === "object" ? value : {};
  const durationMode = ["frame", "frames", "帧", "frame_count"].includes(String(seg.duration_mode || "").toLowerCase()) ? "frames" : "seconds";
  const normalized = {
    prompt: String(seg.prompt || ""),
    duration: durationMode === "frames" ? 5 / 24 : (Number(seg.duration) || 5),
    duration_mode: durationMode,
    audio_filename: String(seg.audio_filename || ""),
    audio_restart: !!seg.audio_restart,
    continue_audio: seg.continue_audio !== false,
    continue_video: seg.continue_video !== false,
    references: Array.isArray(seg.references) ? seg.references.map((ref) => {
      if (typeof ref === "string") return { type: "image", name: ref, path: ref };
      const normalized = { ...ref };
      // ``duration`` is source-media metadata only. Explicit insert fields
      // control the H3 timeline guide; 0/0 preserves reference-only behavior.
      if (normalized.type === "image") delete normalized.duration;
      else normalized.duration = Number(ref?.duration) > 0 ? Number(ref.duration) : 1;
      normalized.insert_seconds = Math.max(0, Number(ref?.insert_seconds) || 0);
      normalized.insert_frames = Math.max(0, Math.floor(Number(ref?.insert_frames) || 0));
      return normalized;
    }) : [],
  };
  if (durationMode === "frames") normalized.frame_count = Math.max(5, Math.floor(Number(seg.frame_count) || 5));
  if (Object.prototype.hasOwnProperty.call(seg, "use_previous_video_reference")) {
    normalized.use_previous_video_reference = !!seg.use_previous_video_reference;
  }
  return normalized;
}

function segmentLengthValid(seg) {
  if (seg?.duration_mode === "frames") return Number(seg.frame_count) >= 5;
  return Number(seg?.duration) >= 1 && Number(seg?.duration) <= 15;
}

function setSegmentLengthMode(seg, mode) {
  if (mode === "frames") {
    seg.duration_mode = "frames";
    seg.frame_count = 5;
    seg.duration = 5 / 24;
  } else {
    seg.duration_mode = "seconds";
    delete seg.frame_count;
    // Restore a practical seconds value when leaving the fixed 5-frame mode.
    if (!(Number(seg.duration) >= 1)) seg.duration = 5;
  }
}

function normalizeSegments(value) {
  return Array.isArray(value) ? value.map(normalizeSegment) : [];
}

function readSegments(node) {
  const w = widget(node, "segments_json");
  try {
    const value = JSON.parse(w?.value || "[]");
    return normalizeSegments(value);
  } catch (_) {
    return [];
  }
}

function writeSegments(node, segments) {
  const w = widget(node, "segments_json");
  if (!w) return;
  const saved = segments.map((segment) => {
    const copy = { ...segment };
    delete copy._references_open;
    delete copy._media_references_open;
    delete copy._audio_references_open;
    return copy;
  });
  w.value = JSON.stringify(saved, null, 2);
  w.callback?.(w.value);
  const globalAssets = widget(node, "global_assets_json");
  if (globalAssets) {
    globalAssets.value = JSON.stringify(segments[0]?.references || [], null, 2);
    globalAssets.callback?.(globalAssets.value);
  }
  syncSerializedWidgets(node);
  node.setDirtyCanvas(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

function cleanSubfolder(value) {
  return String(value || "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "").replace(/\.\./g, "");
}

function filePath(ref) {
  return String(ref.path || ref.name || "").replace(/^input\//, "");
}

function mediaUrl(ref) {
  const path = filePath(ref);
  const parts = path.split("/");
  const filename = parts.pop() || "";
  return `/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(parts.join("/"))}&type=input`;
}

function formatDuration(value) {
  return Number.isFinite(Number(value)) && Number(value) > 0 ? `${Number(value).toFixed(2)} 秒` : "读取时长中…";
}

async function selectFilesWithPython(type, initialDir, useDefaultPath) {
  const response = await fetch("/h3_auto_director/select_files", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, initial_dir: initialDir || "", use_default_path: !!useDefaultPath }),
  });
  let result = {};
  try { result = await response.json(); } catch (_) { /* handled below */ }
  if (!response.ok) throw new Error(result.error || `Python文件选择失败（${response.status}）`);
  return Array.isArray(result.files) ? result.files : [];
}

async function selectDirectoryWithPython(initialDir) {
  const response = await fetch("/h3_auto_director/select_directory", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ initial_dir: initialDir || "" }),
  });
  let result = {};
  try { result = await response.json(); } catch (_) { /* handled below */ }
  if (!response.ok) throw new Error(result.error || `Python目录选择失败（${response.status}）`);
  return String(result.directory || "");
}

async function probeVideoAudio(path) {
  const response = await fetch("/h3_auto_director/video_info", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  let result = {};
  try { result = await response.json(); } catch (_) { /* handled below */ }
  if (!response.ok) throw new Error(result.error || `视频音轨检测失败（${response.status}）`);
  // Keep an unknown probe result distinct from a confirmed silent video.
  // Older plans and environments without ffprobe must still expose the
  // per-video audio toggle instead of hiding it permanently.
  if (result.has_audio === true) return true;
  if (result.has_audio === false) return false;
  return null;
}

async function probeVideoInfo(path) {
  const response = await fetch("/h3_auto_director/video_info", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }),
  });
  let result = {};
  try { result = await response.json(); } catch (_) { /* handled below */ }
  if (!response.ok) throw new Error(result.error || `视频信息读取失败（${response.status}）`);
  return result;
}

async function uploadOne(file, type) {
  const form = new FormData();
  form.append("image", file, file.name);
  form.append("type", "input");
  // The picker directory is only a browsing preference; uploads stay in the
  // stable input folders consumed by the Python node.
  form.append("subfolder", UPLOAD_DIRS[type]);
  form.append("overwrite", "false");
  const response = await fetch("/upload/image", { method: "POST", body: form });
  if (!response.ok) throw new Error(`${file.name} 上传失败（${response.status}）`);
  const result = await response.json();
  const path = [result.subfolder, result.name].filter(Boolean).join("/");
  const ref = { type, name: result.name, path, originalName: file.name };
  if (type === "video") {
    const hasAudio = await probeVideoAudio(path).catch(() => null);
    if (hasAudio !== null) ref.has_audio = hasAudio;
    ref.video_audio_enabled = true;
  }
  return ref;
}

// Keep the standalone video loader compatible with VHS_LoadVideo's native
// uploader.  In particular, use ComfyUI's apiURL/auth handling and the exact
// multipart field name expected by /upload/image instead of fetch()'s custom
// request path.
async function uploadVideoWithComfyUI(file, node) {
  const body = new FormData();
  const relative = String(file.webkitRelativePath || "");
  const slash = relative.lastIndexOf("/");
  const subfolder = slash > 0 ? relative.slice(0, slash + 1) : "";
  const upload = new File([file], file.name, { type: file.type, lastModified: file.lastModified });
  body.append("image", upload);
  if (subfolder) body.append("subfolder", subfolder);
  const response = await new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.upload.onprogress = (event) => { if (event.lengthComputable) node.progress = event.loaded / event.total; };
    request.onload = () => resolve(request);
    request.onerror = () => reject(new Error("视频上传网络连接失败"));
    request.open("POST", api.apiURL("/upload/image"), true);
    Promise.resolve(api.getAuthStore?.()).then(async (store) => {
      const headers = store ? await store.getAuthHeader() : {};
      Object.entries(headers || {}).forEach(([key, value]) => request.setRequestHeader(key, value));
      request.send(body);
    }).catch(reject);
  });
  node.progress = undefined;
  if (response.status !== 200) {
    let detail = `${response.status} ${response.statusText}`;
    if (response.status === 413) detail += "：文件超过 ComfyUI 的上传大小限制，请提高 --max-upload-size 或先放入 input 目录";
    throw new Error(`视频上传失败（${detail}）`);
  }
  const result = JSON.parse(response.responseText || "{}");
  return [result.subfolder, result.name].filter(Boolean).join("/");
}

function countRefs(segment, type) {
  return (segment.references || []).filter((ref) => ref.type === type).length;
}

function totalRefs(segment) {
  return (segment.references || []).length;
}

function videoAudioRefs(segment) {
  return (segment.references || []).filter((ref) => ref.type === "video" && ref.has_audio !== false && ref.video_audio_enabled !== false);
}

function segmentHasContent(segment) {
  return Boolean(String(segment?.prompt || "").trim()) || totalRefs(segment) > 0;
}

function confirmSegmentReduction(segments, target, removedSegments = null) {
  if (target >= segments.length) return true;
  const removed = removedSegments || segments.slice(target);
  const occupied = removed.filter(segmentHasContent).length;
  if (!occupied) return true;
  return window.confirm(`将减少到 ${target} 个片段，并删除 ${occupied} 个包含提示词或参考素材的片段。\n此操作只会在点击保存后写入，是否继续？`);
}

function makeButton(text, handler, title = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = text;
  button.title = title;
  button.onclick = (event) => { event.preventDefault(); handler(event); };
  return button;
}

function removeEmptyReferenceSocket(node) {
  const input = node.inputs?.at(-1);
  if (!input || input.link != null || String(input.name || "").trim()) return;
  node.removeInput(node.inputs.length - 1);
  node.setDirtyCanvas(true, true);
}

function normalizeSaveFps(node) {
  if ((node.comfyClass || node.type) !== SAVE_NODE) return;
  const fps = widget(node, "fps");
  if (fps && (!Number.isFinite(Number(fps.value)) || Number(fps.value) <= 0)) {
    fps.value = 24;
    fps.callback?.(fps.value);
  }
}

function decorateVideoLoad(node) {
  const nodeClass = node.comfyClass || node.type;
  if (![VIDEO_LOAD_NODE, VIDEO_STREAM_NODE].includes(nodeClass) || widget(node, "upload_video")) return;
  // Migrate old workflows that stored a project/final directory.  The node
  // now requires one concrete video file and the user can upload it directly.
  const pathWidget = widget(node, "video_path");
  if (pathWidget && String(pathWidget.value || "").trim()
      && !/\.(mp4|mkv|webm|mov|avi)$/i.test(String(pathWidget.value).trim())) {
    pathWidget.value = "";
    pathWidget.callback?.(pathWidget.value);
  }
  const button = node.addWidget("button", "upload_video", "上传视频", async () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/mp4,video/matroska,video/webm,video/quicktime,video/x-msvideo,.mp4,.mkv,.webm,.mov,.avi";
    input.multiple = false;
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      button.label = "上传中…";
      try {
        const uploadedPath = await uploadVideoWithComfyUI(file, node);
        const path = widget(node, "video_path");
        if (!path) throw new Error("找不到视频文件输入框");
        path.value = uploadedPath;
        path.callback?.(path.value);
        node.setDirtyCanvas?.(true, true);
        button.label = "重新上传视频";
      } catch (error) {
        button.label = "上传视频";
        window.alert(error?.message || String(error));
      }
    };
    input.click();
  });
  button.serialize = false;
  button.label = "上传视频";
  button.tooltip = "直接上传一个视频文件；此节点不接受目录。";
}

function applyChineseLabels(node) {
  const nodeClass = node.comfyClass || node.type;
  if (!H3_NODE_CLASSES.has(nodeClass)) return;
  const labels = {
    project: "项目计划", plan: "项目计划", project_id: "总文件夹名称", segments_json: "片段配置", duration: "默认片段时长",
    global_reference_set: "统一参考集", auto_run: "自动连续生成", continuation_mode: "接续模式", auto_context_crop_frames: "自动裁剪上下文帧数",
    decode_after_all_segments: "所有片段完成后统一解码（逐段处理）",
    skip_reference_encoding: "不编码参考素材",
    output_filename: "统一输出文件名（留空使用 H3）", overwrite_existing: "是否覆盖已有文件",
    context_method: "上下文方案",
    enable_audio_drive: "启用音频驱动", audio_drive_file: "音频驱动文件",
    cache_prompt_embeddings: "一次性缓存提示词向量", cache_prompt_embeddings_to_disk: "缓存提示词向量到硬盘", global_assets_json: "统一参考素材",
    keep_model_loaded: "模型常驻显存（防反复装卸）",
    segment_index: nodeClass === SEGMENT_NODE || nodeClass === CONTEXT_NODE || nodeClass === RESUME_NODE ? "上下文片段序号" : "片段序号",
    context_length: "上下文长度", prompt: "提示词", references_json: "参考素材 JSON",
    clip: "文本编码器", vae: "视频 VAE", audio_vae: "音频 VAE", width: "宽度", height: "高度", length: "帧数",
    ref_image_size: "预设选项（match/max）", use_auto_ref_image_size: "使用预设", use_manual_ref_short_edge: "使用手动设置", ref_short_edge: "参考图最短边", enable_resume: "启用断点续接", latent_path: nodeClass === LOAD_SAVED_AV_LATENT_NODE ? "保存的 AV 潜空间文件路径" : "缓存潜变量路径", video_path: [VIDEO_LOAD_NODE, VIDEO_STREAM_NODE].includes(nodeClass) ? "视频文件（可上传，不能填目录）" : "缓存视频路径",
    conditioning: "条件", latent: "潜变量", context_frames: "上下文画面", context_latent: "上下文潜变量",
    use_video_context: "使用视频上下文", use_audio_context: "使用音频上下文", use_video_latent: "使用视频潜空间",
    fps: "帧率", images: "视频画面", use_stage1_audio_only: "最终仅使用一采音频",
    context_sampled_start_tokens: "首部可采样 latent token 数", context_sampled_start_strength: "首部 token 重绘强度", context_sampled_tokens: "末端可采样 latent token 数", context_sampled_strength: "末端 token 重绘强度",
    audio: "音频", saved_video: nodeClass === "H3AutoDirectorTTSController" ? "已保存音频" : "已保存视频", segment_node_id: "片段节点 ID", trim_frames: "上下文裁剪帧数", auto_context_crop: "自动裁剪上下文", match_tail: "匹配音频尾部",
    clip_index: "片段序号", latent_path: nodeClass === LOAD_SAVED_AV_LATENT_NODE ? "保存的 AV 潜空间文件路径" : "潜变量路径",
    aspect_ratio: "宽高比", megapixels: "目标像素数（MP）", multiple: "尺寸倍数",
    use_preset_ratio: "使用预设比例", use_custom_ratio: "使用自定义比例", aspect_preset: "宽高比预设", custom_ratio: "自定义比例（宽,高）",
    stage1_megapixels: "第一阶段像素数（MP）", stage2_megapixels: "第二阶段像素数（MP）",
    resolution_preview: "当前输出分辨率",
    input_fps: "原视频帧率", interpolation_multiplier: "补帧倍率", vfi_model: "补帧模型", sr_frame_count: "超分处理帧数", sr_scale: "超分倍率（相对原视频）", sr_quality: "RTX VSR 质量", filename: "输出文件名", filename_prefix: "输出文件名前缀", preserve_audio: "保留原视频音频",
    output_root: "项目文件夹名称（保存于 output/h3_project 下）",
    video_format: "视频格式", video_codec: "编码格式", encoder_device: "编码设备", quality: "编码质量", latent_directory: "潜空间目录（项目目录或 cache）", output_intermediate: "输出中间片段", intermediate_filename: "中间片段文件名前缀", final_filename: "最终视频文件名", auto_crop_frames: "自动裁剪帧数（从第2段开始）", color_correction: "上下文色彩校正", resolution_mode: "控制预处理分辨率", target_width: "生成画布宽度", target_height: "生成画布高度", generation_width: "第一阶段宽度", generation_height: "第一阶段高度", target_short_edge: "参考图最短边",
    scene_cut_protection: "场景切换保护", scene_cut_threshold: "场景切换阈值",
    correction_strength: "校色强度", residual_strength: "残余漂移强度",
    cleanup_after_final: "最终完成后清理显存", sampling_mode: "音频采样切换", audio_sampling: "音频采样方法", scheduler: "调度器", steps: "采样步数", denoise: "降噪",
    stage1_steps: "第一阶段步数", stage1_denoise: "第一阶段降噪", enable_stage2: "启用第二阶段采样", stage2_use_context: "开启二采上下文接续", stage2_steps: "第二阶段步数", stage2_denoise: "第二阶段降噪",
    stage1_sigmas: "一采 Sigmas 调度", stage2_sigmas: "二采 Sigmas 调度",
    stage1_extend_sigmas: "一采插值扩展 Sigmas", stage1_extend_steps: "一采插值步数", stage1_start_at_sigma: "一采起始 Sigma", stage1_end_at_sigma: "一采结束 Sigma", stage1_spacing: "一采间距方式",
    stage2_extend_sigmas: "二采插值扩展 Sigmas", stage2_extend_steps: "二采插值步数", stage2_start_at_sigma: "二采起始 Sigma", stage2_end_at_sigma: "二采结束 Sigma", stage2_spacing: "二采间距方式",
    upscale_mode: "视频放大方式", target_width: "第二阶段宽度", target_height: "第二阶段高度", upscale_model: "普通放大模型", latent_upscale_model: "H3 latent 学习型放大模型", latent_upscale_device: "latent 放大设备", latent_upscale_precision: "latent 放大精度", enable_preview: "新版采样预览", seed: "双采样种子",
    shift_video: "视频调度偏移", shift_audio: "音频调度偏移",
    unet_name: nodeClass === HYBRID_LOADER_NODE || nodeClass === TRANSFER_LOADER_NODE ? "多模态参考模型（Ref2VA）" : "扩散模型",
    stage1_model: "一采多模态参考模型（Ref2VA）", stage2_model: "二采多模态参考模型（Ref2VA）", stage1_enable_hybrid: "一采启用 H3 混合模型", stage1_base_model: "一采画面基础模型（FL2VA）", stage2_enable_hybrid: "二采启用 H3 混合模型", stage2_base_model: "二采画面基础模型（FL2VA）",
    base_model: "画面基础模型（FL2VA）", enable_hybrid: "启用 H3 混合模型", weight_dtype: "权重数据类型",
    reference_video_json: "参考动作视频", reference_assets_json: "附加参考素材",
    segment_seconds: "每段秒数", use_reference_video_material: "将上传视频作为多模态参考素材", pass_reference_video_audio: "传递参考视频音频",
    enable_audio_continuation: "开启音频上下文接续", audio_restart_segments: "重新生成音频片段",
    previous_video_reference_segments: "使用上段视频参考片段", skip_h3_audio_decode: "仅不解码 H3 音频（仍联合采样）",
    final_audio_source: "最终视频音频来源", edit_transfer: "编辑动作迁移计划",
    concat_final_audio: "拼接最终长音频", edit_tts: "编辑 TTS 片段",
    control_mode: "控制模式", control_type: "控制类型", input_style: "输入风格", preprocess_device: "预处理设备", enabled: nodeClass === CONTROL_EXPORT_NODE ? "导出控制视频" : "启用预处理",
    save_preprocessed: "保存预处理视频", preprocess_all_segments: "一次性预处理全部片段", pose_weight: "姿态控制权重", depth_weight: "深度控制权重",
    pose_preprocess_type: "姿态预处理类型", pose_model_profile: "姿态预处理模型", depth_model_profile: "深度预处理模型",
    enable_preprocess_chunking: "启用分块预处理", preprocess_chunk_frames: "每批预处理帧数",
    video_frames: "视频帧", source_video_path: "视频路径（可选）", control_video: "控制视频",
    control_context_scale: "控制强度", backend: "Union 后端", controlnet_path: "Union 权重路径",
    config: "控制配置", control_config: "姿态/深度控制配置", output_name: "输出文件名",
  };
  const apply = (item) => {
    const stageLabel = nodeClass === DUAL_SAMPLING_NODE
      ? { stage1_model: "一采模型（可接外部 LoRA/显存优化）", stage2_model: "二采模型（未连接复用一采）" }[item?.name]
      : nodeClass === SAMPLING_SWITCH_NODE && item?.name === "model"
        ? "模型（仅用于原版 SIGMAS）"
      : null;
    const label = stageLabel || labels[item?.name];
    if (!label) return;
    item.label = label;
    item.localized_name = label;
    item.widget && (item.widget.label = label, item.widget.localized_name = label);
  };
  (node.widgets || []).forEach(apply);
  (node.inputs || []).forEach(apply);
}

function decorateNode(node) {
  const nodeClass = node.comfyClass || node.type;
  if (!H3_NODE_CLASSES.has(nodeClass)) return;
  if (nodeClass === DUAL_STAGE_LOADER_NODE) cleanDualStageLoaderPorts(node);
  if (nodeClass === SAMPLING_SWITCH_NODE) cleanSamplingSwitchPorts(node);
  if (nodeClass === CONTROL_PREPROCESS_NODE) removeRetiredPorts(node, ["source_video_path", "视频路径（可选）"], []);
  if (nodeClass === CONTROLLER_NODE) removeRetiredPorts(node, ["crop_context_on_assemble", "拼接前裁剪上下文", "output_root", "输出文件名（最终视频，留空使用 H3）"]);
  if (nodeClass === SAVE_NODE) removeRetiredPorts(node, ["output_root", "输出文件名（中间片段，留空使用 H3）"]);
  if (nodeClass === CONTEXT_NODE) {
    removeRetiredPorts(node, ["context_stage"]);
    if (!(node.outputs || []).some((output) => output?.name === "二采上下文潜变量")) {
      node.addOutput?.("二采上下文潜变量", "LATENT");
    }
    if (node.outputs?.[0]) node.outputs[0].name = "上下文画面";
    if (node.outputs?.[1]) node.outputs[1].name = "上下文潜变量";
    if (node.outputs?.[2]) node.outputs[2].name = "二采上下文潜变量";
  }
  if (nodeClass === MOTION_CONTEXT_NODE) {
    const methodWidget = widget(node, "context_method");
    if (methodWidget) {
      methodWidget.options = methodWidget.options || {};
      methodWidget.options.values = ["潜空间直取"];
      methodWidget.value = "潜空间直取";
    }
  }
  if (nodeClass === NODE && !widget(node, "edit_segments")) {
    const button = node.addWidget("button", "edit_segments", "编辑片段", () => openEditor(node));
    button.label = "编辑片段";
    button.serialize = false;
  }
  if (nodeClass === TRANSFER_NODE && !widget(node, "edit_transfer")) {
    const button = node.addWidget("button", "edit_transfer", "编辑动作迁移计划", () => openTransferEditor(node));
    button.label = "编辑动作迁移计划";
    button.serialize = false;
  }
  if (nodeClass === TTS_NODE && !widget(node, "edit_tts")) {
    const button = node.addWidget("button", "edit_tts", "编辑 TTS 片段", () => openTTSPlanEditor(node));
    button.label = "编辑 TTS 片段";
    button.serialize = false;
  }
  if (nodeClass === SEGMENT_NODE && !widget(node, "reset_segment_index")) {
    const button = node.addWidget("button", "reset_segment_index", "↺ 重置为第 1 段 (序号 0)", () => {
      const segW = widget(node, "segment_index");
      if (segW) {
        segW.value = 0;
        segW.callback?.(0);
        node.setDirtyCanvas?.(true, true);
        node.graph?.setDirtyCanvas?.(true, true);
      }
    });
    button.label = "↺ 重置为第 1 段 (序号 0)";
    button.serialize = false;
  }
  if (nodeClass === TRANSFER_NODE && !widget(node, "use_reference_video_material")) {
    const material = node.addWidget("toggle", "use_reference_video_material", "将上传视频作为多模态参考素材", true, (value) => {
      material.value = !!value;
      node.setDirtyCanvas?.(true, true);
    });
    material.label = "将上传视频作为多模态参考素材";
    material.serialize = true;
  }
  if (nodeClass === H3_RESOLUTION_NODE) decorateH3Resolution(node);
  if (nodeClass === CACHED_REFERENCE_NODE) decorateCachedReference(node);
  if ([VIDEO_LOAD_NODE, VIDEO_STREAM_NODE].includes(nodeClass)) decorateVideoLoad(node);
  const labels = {
    project_id: "总文件夹名称",
    plan: "项目计划",
    segments_json: "片段配置",
    duration: "默认片段时长",
    global_reference_set: "统一参考集",
    auto_run: "自动连续生成",
    continuation_mode: "接续模式",
    auto_context_crop_frames: "自动裁剪上下文帧数",
    decode_after_all_segments: "所有片段完成后统一解码（逐段处理）",
    skip_reference_encoding: "不编码参考素材",
    cache_prompt_embeddings: "一次性缓存提示词向量",
    cache_prompt_embeddings_to_disk: "缓存提示词向量到硬盘",
    keep_model_loaded: "模型常驻显存（防反复装卸）",
    global_assets_json: "统一参考素材",
    output_root: nodeClass === SAVE_NODE ? "输出文件名（中间片段，留空使用 H3）" : nodeClass === CONTROLLER_NODE ? "输出文件名（最终视频，留空使用 H3）" : "项目文件夹名称（保存于 output/h3_project 下）",
    output_filename: "统一输出文件名（留空使用 H3）",
    overwrite_existing: "覆盖已有文件",
    context_method: "上下文方案",
    stage2_use_context: "开启二采上下文接续",
    video_format: "视频格式",
    video_codec: "编码格式",
    encoder_device: "编码设备",
    quality: "编码质量",
    video_path: [VIDEO_LOAD_NODE, VIDEO_STREAM_NODE].includes(nodeClass) ? "视频文件（可上传，不能填目录）" : "输入视频文件路径", input_fps: "原视频帧率", interpolation_multiplier: "补帧倍率", vfi_model: "补帧模型", sr_frame_count: "超分处理帧数", sr_scale: "超分倍率（相对原视频）", sr_quality: "RTX VSR 质量", filename: "输出文件名", filename_prefix: "输出文件名前缀", preserve_audio: "保留原视频音频",
    latent_directory: "潜空间目录（项目目录或 cache）", output_intermediate: "输出中间片段", intermediate_filename: "中间片段文件名前缀", final_filename: "最终视频文件名", auto_crop_frames: "自动裁剪帧数（从第2段开始）",
    color_correction: "上下文色彩校正",
    use_video_latent: "使用视频潜空间",
    context_sampled_start_tokens: "首部可采样 latent token 数", context_sampled_start_strength: "首部 token 重绘强度", context_sampled_tokens: "末端可采样 latent token 数", context_sampled_strength: "末端 token 重绘强度",
    scene_cut_protection: "场景切换保护",
    scene_cut_threshold: "场景切换阈值",
    correction_strength: "校色强度",
    residual_strength: "残余漂移强度",
    cleanup_after_final: "最终完成后清理显存",
    sampling_mode: "音频采样切换",
    shift_video: "视频调度偏移",
    shift_audio: "音频调度偏移",
    reference_video_json: "参考视频素材",
    reference_assets_json: "参考素材",
    segment_seconds: "每段秒数",
    pass_reference_video_audio: "传递参考视频音频",
    enable_audio_continuation: "开启音频上下文接续",
    audio_restart_segments: "重新生成音频片段",
    previous_video_reference_segments: "使用上段视频参考片段",
    skip_h3_audio_decode: "仅不解码 H3 音频（仍联合采样）",
    final_audio_source: "最终视频音频来源",
    concat_final_audio: "拼接最终长音频",
    edit_tts: "编辑 TTS 片段",
  };
  applyChineseLabels(node);
  if (nodeClass === SAVE_NODE || nodeClass === CONTROLLER_NODE) {
    const output = widget(node, "output_root");
    if (output) {
      if (nodeClass === SAVE_NODE && String(output.value || "").trim() === "h3_projects") {
        output.value = "";
        output.callback?.(output.value);
      }
      output.label = "输出文件名（可选，留空沿用计划设置）";
    }
  }
  normalizeSaveFps(node);
}

function decorateCachedReference(node) {
  const auto = widget(node, "use_auto_ref_image_size");
  const manual = widget(node, "use_manual_ref_short_edge");
  const edge = widget(node, "ref_short_edge");
  const mode = widget(node, "ref_image_size");
  const nearest32 = (value) => Math.max(32, Math.round(Number(value || 2048) / 32) * 32);
  const boolValue = (value) => !(typeof value === "string" && ["false", "0", "off", "关闭"].includes(value.trim().toLowerCase()));
  let normalizing = false;
  let hydrated = false;
  // LiteGraph can briefly expose the old toggle value while a widget callback
  // is running. Keep the user's last mode separately so editing the numeric
  // short-edge field cannot accidentally fall back to preset mode.
  const storedMode = node.properties?.h3_ref_sizing_mode;
  let selectedMode = (storedMode === "preset" || storedMode === "manual")
    ? storedMode : (node.__h3CachedReferenceSizingMode || null);
  const setWidgetBoolean = (item, value) => {
    if (!item) return;
    const changed = boolValue(item.value) !== !!value;
    item.value = !!value;
    if (item.options) item.options.value = !!value;
    if (changed && typeof item.callback === "function" && !normalizing) item.callback(!!value);
  };
  const update = (preferredMode = null) => {
    // Older saved workflows may not contain the newly added widgets. Always
    // normalize both values before drawing so there is never a dead state.
    let presetOn = boolValue(auto?.value);
    let manualOn = boolValue(manual?.value);
    if (preferredMode === "preset") {
      selectedMode = "preset";
      presetOn = true; manualOn = false;
    } else if (preferredMode === "manual") {
      selectedMode = "manual";
      presetOn = false; manualOn = true;
    } else if (selectedMode === "preset") {
      presetOn = true; manualOn = false;
    } else if (selectedMode === "manual") {
      presetOn = false; manualOn = true;
    } else if (presetOn === manualOn) {
      // An equal pair can be the transient default values shown before a
      // saved workflow's widgets are hydrated. Do not persist a mode yet.
      if (hydrated) selectedMode = "preset";
      presetOn = true; manualOn = false;
    } else if (hydrated) {
      // Hydrate the remembered mode from an existing, valid pair of widget
      // values before any later numeric-field callback can run.
      selectedMode = presetOn ? "preset" : "manual";
    }
    node.__h3CachedReferenceSizingMode = selectedMode;
    if (selectedMode) {
      node.properties = node.properties || {};
      node.properties.h3_ref_sizing_mode = selectedMode;
    }
    normalizing = true;
    if (auto) { auto.value = presetOn; if (auto.options) auto.options.value = presetOn; }
    if (manual) { manual.value = manualOn; if (manual.options) manual.options.value = manualOn; }
    normalizing = false;
    if (auto) auto.hidden = false;
    if (mode) mode.hidden = !presetOn;
    if (edge) {
      edge.hidden = presetOn;
      if (!presetOn) {
        const aligned = nearest32(edge.value);
        if (Number(edge.value) !== aligned) {
          edge.value = aligned;
          edge.callback?.(aligned);
        }
      }
    }
    if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
      const computed = node.computeSize();
      const currentW = Array.isArray(node.size) ? node.size[0] : 0;
      const currentH = Array.isArray(node.size) ? node.size[1] : 0;
      const savedW = node.properties?.h3_node_size?.[0] || 0;
      const savedH = node.properties?.h3_node_size?.[1] || 0;
      const targetW = Math.max(currentW, computed[0], savedW, 350);
      const targetH = Math.max(currentH, computed[1], savedH, 310);
      if (currentW < targetW || currentH < targetH) {
        node.setSize([targetW, targetH]);
      }
    }
    node.setDirtyCanvas(true, true);
  };
  if (!node.__h3CachedReferenceSizingBound) {
    node.__h3CachedReferenceSizingBound = true;
    const origResize = node.onResize;
    node.onResize = function (size) {
      const res = origResize?.apply(this, arguments);
      if (Array.isArray(size)) {
        this.properties = this.properties || {};
        this.properties.h3_node_size = [size[0], size[1]];
      }
      return res;
    };
    const switchMode = (source, modeName, value, args, previous) => {
      // LiteGraph versions differ in when they assign widget.value. Prefer
      // the callback argument when present; it is the newly clicked state.
      const enabled = value === undefined ? boolValue(source?.value) : boolValue(value);
      if (source) source.value = enabled;
      const selectedMode = enabled ? modeName : (modeName === "preset" ? "manual" : "preset");
      const result = previous?.call(source, enabled, ...args);
      update(selectedMode);
      return result;
    };
    if (auto) {
      const previous = auto.callback;
      auto.callback = function (value, ...args) {
        return switchMode(this, "preset", value, args, previous);
      };
    }
    if (manual) {
      const previous = manual.callback;
      manual.callback = function (value, ...args) {
        return switchMode(this, "manual", value, args, previous);
      };
    }
    if (edge) {
      const previous = edge.callback;
      edge.callback = function (value, ...args) {
        const aligned = nearest32(value);
        this.value = aligned;
        const result = previous?.call(this, aligned, ...args);
        update(selectedMode);
        return result;
      };
    }
  }
  update();
  // Node widgets can be hydrated one frame after loadedGraphNode. Reapply
  // the invariant after hydration as well as on the initial decoration.
  requestAnimationFrame(() => { hydrated = true; update(); });
  setTimeout(() => { hydrated = true; update(); }, 120);
}

function h3ResolutionValue(node, name, fallback) {
  const value = widget(node, name)?.value;
  return value === undefined || value === null || value === "" ? fallback : value;
}

function h3BooleanValue(value, fallback = false) {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "string") return !["false", "0", "off", "关闭"].includes(value.trim().toLowerCase());
  return Boolean(value);
}

function calculateH3Resolution(ratioWidth, ratioHeight, megapixels, multiple) {
  const ratio = Math.max(1, Number(ratioWidth) || 1) / Math.max(1, Number(ratioHeight) || 1);
  // MiniMax H3 requires a 32px canvas grid.  Keep preview calculations in
  // lockstep with the Python node even when an old workflow stores 16/24.
  const requestedMultiple = Math.max(32, Math.trunc(Number(multiple) || 32));
  const alignment = Math.max(32, Math.floor(requestedMultiple / 32) * 32);
  const targetPixels = Math.max(0.2, Math.min(5, Number(megapixels) || 0.2)) * 1024 * 1024;
  let width = Math.max(alignment, Math.round(Math.sqrt(targetPixels * ratio) / alignment) * alignment);
  let height = Math.max(alignment, Math.round((Math.sqrt(targetPixels * ratio) / ratio) / alignment) * alignment);
  const maxDimension = Math.max(alignment, Math.floor(16384 / alignment) * alignment);
  if (Math.max(width, height) > maxDimension) {
    const scale = maxDimension / Math.max(width, height);
    width = Math.max(alignment, Math.round(width * scale / alignment) * alignment);
    height = Math.max(alignment, Math.round(height * scale / alignment) * alignment);
  }
  return { width, height, megapixels: (width * height / 1000000).toFixed(2) };
}

function decorateH3Resolution(node) {
  const presets = { "16:9": [16, 9], "9:16": [9, 16], "1:1": [1, 1], "4:3": [4, 3], "3:4": [3, 4], "3:2": [3, 2], "2:3": [2, 3], "21:9": [21, 9] };
  let preview = widget(node, "resolution_preview");
  if (!preview) {
    preview = node.addWidget("text", "resolution_preview", "", () => {});
    preview.label = "当前输出分辨率";
    preview.serialize = false;
    preview.options = { multiline: true };
  }
  const updatePreview = () => {
    const presetEnabled = h3BooleanValue(h3ResolutionValue(node, "use_preset_ratio", true), true);
    const customEnabled = h3BooleanValue(h3ResolutionValue(node, "use_custom_ratio", false), false);
    const preset = String(h3ResolutionValue(node, "aspect_preset", "16:9"));
    const ratioText = String(h3ResolutionValue(node, "custom_ratio", "16,9")).replace(/，/g, ",");
    const custom = ratioText.split(",").map((part) => Number(part.trim()));
    const ratio = customEnabled && custom.length === 2 && custom.every((value) => Number.isFinite(value) && value > 0)
      ? custom : (presets[preset] || [16, 9]);
    const first = calculateH3Resolution(ratio[0], ratio[1], h3ResolutionValue(node, "stage1_megapixels", 0.4), h3ResolutionValue(node, "multiple", 32));
    const second = calculateH3Resolution(ratio[0], ratio[1], h3ResolutionValue(node, "stage2_megapixels", 0.98), h3ResolutionValue(node, "multiple", 32));
    const multipleWidget = widget(node, "multiple");
    if (multipleWidget && Number(multipleWidget.value) < 32) {
      multipleWidget.value = 32;
      multipleWidget.callback?.(32);
    }
    const selectedText = customEnabled ? `${ratio[0]}:${ratio[1]}` : (presetEnabled ? preset : "16:9");
    preview.value = `第一阶段：${first.width} x ${first.height}（${first.megapixels} MP）\n第二阶段：${second.width} x ${second.height}（${second.megapixels} MP）\n比例：${selectedText}`;
    const presetWidget = widget(node, "aspect_preset");
    const customWidget = widget(node, "custom_ratio");
    if (presetWidget) presetWidget.hidden = customEnabled || !presetEnabled;
    if (customWidget) customWidget.hidden = !customEnabled;
    node.setDirtyCanvas(true, true);
  };
  if (!node.__h3ResolutionPreviewBound) {
    node.__h3ResolutionPreviewBound = true;
    ["use_preset_ratio", "use_custom_ratio", "aspect_preset", "custom_ratio", "stage1_megapixels", "stage2_megapixels", "multiple"].forEach((name) => {
      const input = widget(node, name);
      if (!input) return;
      const previous = input.callback;
      input.callback = function (value, ...args) {
        if (name === "use_preset_ratio" && h3BooleanValue(value)) {
          const custom = widget(node, "use_custom_ratio");
          if (custom) { custom.value = false; custom.callback?.(false); }
        } else if (name === "use_custom_ratio" && h3BooleanValue(value)) {
          const preset = widget(node, "use_preset_ratio");
          if (preset) { preset.value = false; preset.callback?.(false); }
        }
        const result = previous?.call(this, value, ...args);
        requestAnimationFrame(updatePreview);
        return result;
      };
    });
  }
  updatePreview();
}

function openTTSPlanEditor(node) {
  const get = (name, fallback) => { const value = widget(node, name)?.value; return value === undefined || value === null ? fallback : value; };
  const set = (name, value) => {
    const item = widget(node, name);
    if (!item) return false;
    // Some ComfyUI 0.34 builds keep a separate serialized values array for
    // hidden/string widgets. Updating only ``item.value`` changes the editor
    // preview but leaves the queued prompt with the previous JSON/path.
    item.value = value;
    item.callback?.(value);
    const index = (node.widgets || []).indexOf(item);
    if (index >= 0) {
      node.widgets_values = Array.isArray(node.widgets_values) ? node.widgets_values : [];
      node.widgets_values[index] = value;
    }
    return true;
  };
  let segments = [], legacyVideo = {}, legacyAssets = [];
  try { segments = JSON.parse(get("segments_json", "[]") || "[]"); } catch (_) { segments = []; }
  try { legacyVideo = JSON.parse(get("reference_video_json", "{}") || "{}"); } catch (_) { legacyVideo = {}; }
  try { legacyAssets = JSON.parse(get("reference_assets_json", "[]") || "[]"); } catch (_) { legacyAssets = []; }
  if (!Array.isArray(segments) || !segments.length) segments = [{ prompt: "", duration: 5, audio_filename: "", continue_audio: true, references: [] }];
  const legacyRefs = [...(Array.isArray(legacyAssets) ? legacyAssets : [])];
  if (legacyVideo?.path || legacyVideo?.name) legacyRefs.push({ ...legacyVideo, type: "video", video_audio_enabled: get("pass_reference_video_audio", false) });
  // Migrate only rows that truly lack the new per-segment field. An explicit
  // empty array is meaningful and must remain empty when unified references
  // are disabled, even though the compatibility field may contain assets.
  segments.forEach((seg) => {
    if (!("references" in seg)) {
      seg.references = legacyRefs.map((ref) => ({ ...ref }));
    }
  });
  const shade = document.createElement("div"); shade.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif";
  const panel = document.createElement("div"); panel.style.cssText = "width:min(1120px,95vw);height:min(860px,92vh);min-width:min(760px,calc(100vw - 24px));min-height:min(520px,calc(100vh - 24px));max-height:92vh;box-sizing:border-box;resize:both;overflow:hidden;background:#20252b;color:#eee;border:1px solid #59636e;border-radius:8px;padding:18px;display:flex;flex-direction:column";
  const title = document.createElement("h2"); title.textContent = "H3 自动导演｜TTS 片段计划"; title.style.margin = "0 0 12px"; panel.appendChild(title);
  const notice = document.createElement("div"); notice.style.cssText = "color:#aeb7c1;font-size:12px;min-height:22px"; panel.appendChild(notice);
  const settings = document.createElement("div"); settings.style.cssText = "display:flex;flex-wrap:wrap;gap:12px;padding:10px;background:#15191d;border:1px solid #424b55;border-radius:6px;margin-bottom:12px"; panel.appendChild(settings);
  const checkbox = (label, name, fallback) => { const wrap = document.createElement("label"); wrap.style.cssText = "display:flex;align-items:flex-start;gap:6px;flex:1 1 240px;min-width:180px;line-height:1.4;white-space:normal;overflow-wrap:anywhere"; const input = document.createElement("input"); input.type = "checkbox"; input.checked = !!get(name, fallback); input.style.cssText = "flex:0 0 auto;margin-top:2px"; const text = document.createElement("span"); text.textContent = label; wrap.append(input, text); settings.appendChild(wrap); return input; };
  const unified = checkbox("统一参考集（所有片段使用第 1 段素材）", "global_reference_set", false);
  const concat = checkbox("拼接最终长音频", "concat_final_audio", true);
  const continuation = checkbox("开启音频上下文接续", "enable_audio_continuation", true);
  const cache = checkbox("一次性缓存提示词向量", "cache_prompt_embeddings", true);
  const diskCache = checkbox("缓存提示词向量到硬盘（关闭一次性缓存时仅处理当前片段）", "cache_prompt_embeddings_to_disk", true);
  const keepModel = checkbox("片段间模型常驻显存（防止反复装卸）", "keep_model_loaded", true);
  const list = document.createElement("div"); list.style.cssText = "display:flex;flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;flex-direction:column;gap:10px;padding:0 6px 10px 0"; panel.appendChild(list);
  const refName = (ref) => ref.originalName || ref.name || ref.path || "未命名素材";
  const refLabel = (ref, type, refs) => { const ordinal = refs.filter((x) => x.type === type).indexOf(ref) + 1; return `${type === "image" ? "图片" : type === "video" ? "视频" : "音频"}${ordinal}：${refName(ref)}`; };
  const addFiles = async (seg, type, renderRefs) => {
    const options = readPickerOptions(); let selected = [];
    if (options.mode === "python") selected = await selectFilesWithPython(type, readDirectories()[type], options.useDefaultPath);
    else { const input = document.createElement("input"); input.type = "file"; input.multiple = true; input.accept = type === "image" ? "image/*" : type === "video" ? "video/*" : "audio/*"; selected = await new Promise((resolve) => { input.onchange = () => resolve(Array.from(input.files || [])); input.click(); }); }
    const refs = seg.references || (seg.references = []);
    if (refs.length + selected.length > MAX_TOTAL_REFS) throw new Error(`每段参考素材最多 ${MAX_TOTAL_REFS} 个`);
    if (refs.filter((x) => x.type === type).length + selected.length > MAX_REFS[type]) throw new Error(`${type === "image" ? "图片" : type === "video" ? "视频" : "音频"}参考最多 ${MAX_REFS[type]} 个`);
    if (type === "video" && selected.length + refs.filter((x) => x.type === "video").length > 3) throw new Error("视频参考最多 3 个");
    for (const item of selected) refs.push(item instanceof File ? await uploadOne(item, type) : item);
    renderRefs();
  };
  const render = () => {
    list.replaceChildren();
    segments.forEach((seg, index) => {
      if (!Array.isArray(seg.references)) seg.references = [];
      const box = document.createElement("div"); box.style.cssText = "padding:10px;background:#15191d;border:1px solid #424b55;border-radius:6px";
      const head = document.createElement("div"); head.textContent = `片段 ${index + 1}`; head.style.fontWeight = "600"; box.appendChild(head);
      const prompt = document.createElement("textarea"); prompt.value = String(seg.prompt || ""); prompt.placeholder = "H3 TTS 完整提示词"; prompt.style.cssText = "width:100%;min-height:70px;box-sizing:border-box;margin-top:7px;background:#20252b;color:#eee;border:1px solid #59636e;padding:7px"; prompt.oninput = () => { seg.prompt = prompt.value; }; box.appendChild(prompt);
      const controls = document.createElement("div"); controls.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:7px;min-width:0";
      const mode = document.createElement("select"); mode.innerHTML = "<option value=\"seconds\">秒数</option><option value=\"frames\">5 帧</option>"; mode.value = seg.duration_mode || "seconds"; mode.onchange = () => { setSegmentLengthMode(seg, mode.value); render(); };
      const duration = document.createElement("input"); duration.type = "number"; duration.min = "1"; duration.max = "15"; duration.step = "0.1"; duration.value = Number(seg.duration) >= 1 ? Number(seg.duration) : 5; duration.style.width = "80px"; duration.disabled = mode.value === "frames"; duration.oninput = () => { seg.duration = Math.max(1, Number(duration.value) || 5); };
      const filename = document.createElement("input"); filename.type = "text"; filename.value = String(seg.audio_filename || ""); filename.placeholder = `H3_${String(index + 1).padStart(5, "0")}.wav`; filename.style.cssText = "width:250px;max-width:100%;box-sizing:border-box;background:#20252b;color:#eee;border:1px solid #59636e;padding:5px"; filename.oninput = () => { seg.audio_filename = filename.value; };
      controls.append(mode, duration, "音频文件名", filename); if (segments.length > 1) controls.appendChild(makeButton("删除片段", () => { segments.splice(index, 1); render(); })); box.appendChild(controls);
      const details = document.createElement("details"); details.open = seg._media_references_open === true; details.ontoggle = () => { seg._media_references_open = details.open; }; details.style.cssText = "margin-top:9px;padding:8px;background:#20252b;border:1px solid #59636e;border-radius:5px"; const summary = document.createElement("summary"); summary.textContent = `图片/视频参考（${seg.references.filter((r) => r.type === "image" || r.type === "video").length}）`; summary.style.cursor = "pointer"; details.appendChild(summary);
      const mediaList = document.createElement("div"); mediaList.style.cssText = "display:flex;flex-direction:column;gap:6px;margin-top:8px";
      const audioDetails = document.createElement("details"); audioDetails.open = seg._audio_references_open !== false; audioDetails.ontoggle = () => { seg._audio_references_open = audioDetails.open; }; audioDetails.style.cssText = "margin-top:8px;padding:8px;background:#20252b;border:1px solid #59636e;border-radius:5px"; const audioSummary = document.createElement("summary"); audioSummary.textContent = `音频参考（${seg.references.filter((r) => r.type === "audio").length}）`; audioSummary.style.cursor = "pointer"; audioDetails.appendChild(audioSummary); const audioList = document.createElement("div"); audioList.style.cssText = "display:flex;flex-direction:column;gap:6px;margin-top:8px"; audioDetails.appendChild(audioList);
      const audioContinuation = document.createElement("label"); audioContinuation.style.cssText = "display:flex;align-items:center;gap:6px;margin-top:8px;font-size:12px;color:#d6dde5"; const audioContinuationInput = document.createElement("input"); audioContinuationInput.type = "checkbox"; audioContinuationInput.checked = seg.continue_audio === false; audioContinuationInput.onchange = () => { seg.continue_audio = !audioContinuationInput.checked; }; audioContinuation.append(audioContinuationInput, "关闭本段音频接续"); box.appendChild(audioContinuation);
      const renderRefs = () => {
        const editable = !unified.checked || index === 0;
        mediaList.replaceChildren(); audioList.replaceChildren();
        if (!editable) { mediaList.append("统一参考集已开启，素材来自第 1 段。"); audioList.append("统一参考集已开启，素材来自第 1 段。"); }
        const refs = unified.checked && index > 0 ? segments[0].references : seg.references;
        refs.forEach((ref) => {
          const type = ref.type; if (type !== "image" && type !== "video" && type !== "audio") return;
          if (type === "video" && ref.path && ref.has_audio === undefined && !VIDEO_AUDIO_PROBES.has(ref)) {
            VIDEO_AUDIO_PROBES.add(ref);
            probeVideoAudio(ref.path).then((hasAudio) => {
              ref.has_audio = hasAudio;
              if (ref.video_audio_enabled === undefined) ref.video_audio_enabled = true;
              renderRefs();
            }).catch(() => { ref.has_audio = null; renderRefs(); });
          }
          const row = document.createElement("div"); row.style.cssText = "display:flex;flex-wrap:wrap;align-items:flex-start;gap:8px;min-width:0;padding:6px;background:#15191d;border:1px solid #424b55;border-radius:4px";
          const refText = document.createElement("span"); refText.textContent = refLabel(ref, type, refs); refText.style.cssText = "flex:1 1 180px;min-width:0;overflow-wrap:anywhere;line-height:1.4"; row.appendChild(refText); if (type === "video") { const toggle = document.createElement("label"); toggle.style.cssText = "display:flex;align-items:flex-start;gap:4px;flex:1 1 220px;min-width:180px;font-size:12px;line-height:1.4;white-space:normal;overflow-wrap:anywhere"; const input = document.createElement("input"); input.type = "checkbox"; input.checked = ref.video_audio_enabled !== false; input.disabled = !editable; input.title = "关闭后只传递视频画面"; input.style.cssText = "flex:0 0 auto;margin-top:2px"; input.onchange = () => { ref.video_audio_enabled = input.checked; renderRefs(); }; const toggleText = document.createElement("span"); toggleText.textContent = ref.has_audio === false ? "传递视频音频（未检测到音轨）" : "传递视频音频"; toggle.append(input, toggleText); row.appendChild(toggle); }
          if (editable) row.appendChild(makeButton("删除", () => { seg.references.splice(seg.references.indexOf(ref), 1); renderRefs(); }));
          (type === "audio" ? audioList : mediaList).appendChild(row);
        });
        summary.textContent = `图片/视频参考（${refs.filter((r) => r.type === "image" || r.type === "video").length}）`; audioSummary.textContent = `音频参考（${refs.filter((r) => r.type === "audio").length}）`;
      };
      details.appendChild(makeButton("+ 添加图片", () => editableOrNotice(unified, index, () => addFiles(seg, "image", renderRefs), notice)));
      details.appendChild(makeButton("+ 添加视频", () => editableOrNotice(unified, index, () => addFiles(seg, "video", renderRefs), notice)));
      details.appendChild(mediaList); audioDetails.appendChild(makeButton("+ 添加音频参考", () => editableOrNotice(unified, index, () => addFiles(seg, "audio", renderRefs), notice))); audioDetails.appendChild(audioList); box.append(details, audioDetails); list.appendChild(box); renderRefs();
    });
  };
  const editableOrNotice = (flag, index, fn, message) => { if (flag.checked && index > 0) { message.textContent = "统一参考集已开启，请编辑第 1 段或关闭统一参考集。"; return; } Promise.resolve(fn()).catch((error) => { message.textContent = error.message || String(error); }); };
  unified.onchange = render;
  const actions = document.createElement("div"); actions.style.cssText = "flex:0 0 auto;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px;min-width:0;margin:14px -18px -18px;padding:12px 18px;background:rgba(32,37,43,.98);border-top:1px solid #59636e";
  actions.append(makeButton("+ 添加片段", () => { segments.push({ prompt: "", duration: 5, duration_mode: "seconds", audio_filename: "", audio_restart: false, continue_audio: continuation.checked, references: [], _media_references_open: false, _audio_references_open: true }); render(); }), makeButton("将第 1 段参考素材应用到全部", () => { const refs = (segments[0].references || []).map((ref) => ({ ...ref })); segments.forEach((seg) => { seg.references = refs.map((ref) => ({ ...ref })); }); unified.checked = true; render(); }), makeButton("取消", () => shade.remove()), makeButton("保存", () => { const names = new Set(); for (const seg of segments) { if (!segmentLengthValid(seg)) { notice.textContent = "每段时长最低 1 秒，或选择有效的 5 帧模式。"; return; } const name = String(seg.audio_filename || "").trim(); if (name && !name.toLowerCase().endsWith(".wav")) { notice.textContent = "音频文件名必须使用 .wav 扩展名。"; return; } if (name && names.has(name.toLowerCase())) { notice.textContent = `音频文件名重复：${name}`; return; } if (name) names.add(name.toLowerCase()); } if (unified.checked) { const refs = (segments[0].references || []).map((ref) => ({ ...ref })); segments.forEach((seg) => { seg.references = refs.map((ref) => ({ ...ref })); }); } const savedSegments = segments.map((seg) => { const copy = { ...seg }; delete copy._media_references_open; delete copy._audio_references_open; return copy; }); set("segments_json", JSON.stringify(savedSegments, null, 2)); set("global_reference_set", unified.checked); set("cache_prompt_embeddings", cache.checked); set("cache_prompt_embeddings_to_disk", diskCache.checked); set("keep_model_loaded", keepModel.checked); set("enable_audio_continuation", continuation.checked); set("concat_final_audio", concat.checked); node.setDirtyCanvas(true, true); shade.remove(); }));
  panel.appendChild(actions); shade.appendChild(panel); document.body.appendChild(shade); render();
}

function openTransferEditor(node) {
  const get = (name, fallback) => {
    const value = widget(node, name)?.value;
    return value === undefined || value === null ? fallback : value;
  };
  let video = {};
  let assets = [];
  try { video = JSON.parse(get("reference_video_json", "{}") || "{}"); } catch (_) { video = {}; }
  try { assets = JSON.parse(get("reference_assets_json", "[]") || "[]"); } catch (_) { assets = []; }
  const set = (name, value) => {
    const item = widget(node, name);
    if (!item) return false;
    item.value = value;
    item.callback?.(value);
    const index = (node.widgets || []).indexOf(item);
    if (index >= 0) {
      node.widgets_values = Array.isArray(node.widgets_values) ? node.widgets_values : [];
      node.widgets_values[index] = value;
    }
    return true;
  };
  const shade = document.createElement("div");
  shade.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(920px,94vw);height:min(760px,90vh);min-width:min(680px,calc(100vw - 24px));min-height:min(480px,calc(100vh - 24px));resize:both;overflow:auto;background:#20252b;color:#eee;border:1px solid #59636e;border-radius:8px;padding:18px;box-shadow:0 16px 60px #000;box-sizing:border-box";
  const title = document.createElement("h2"); title.textContent = "H3 自动导演｜动作迁移项目计划"; title.style.margin = "0 0 12px"; panel.appendChild(title);
  const notice = document.createElement("div"); notice.style.cssText = "color:#aeb7c1;font-size:12px;min-height:22px;margin:6px 0"; panel.appendChild(notice);
  const summary = document.createElement("div"); summary.style.cssText = "padding:10px;background:#15191d;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;line-height:1.6"; panel.appendChild(summary);
  const refreshSummary = () => {
    const seconds = lengthMode.value === "frames" ? 5 / 24 : (Number(get("segment_seconds", 5)) || 5);
    const duration = Number(video.duration || 0);
    const count = duration > 0 ? Math.ceil(duration / seconds) : 0;
    summary.textContent = video.path ? `参考视频：${video.originalName || video.name || video.path}｜时长：${formatDuration(duration)}｜按 ${seconds.toFixed(2)} 秒/段：${count || "等待读取"} 段` : "尚未上传参考视频";
  };
  const row = (label, control) => { const wrap = document.createElement("label"); wrap.style.cssText = "display:flex;flex-wrap:wrap;align-items:flex-start;gap:8px;min-width:0;margin:8px 0;line-height:1.4"; const text = document.createElement("span"); text.textContent = label; text.style.cssText = "flex:1 1 240px;min-width:150px;white-space:normal;overflow-wrap:anywhere"; control.style.maxWidth = "100%"; control.style.boxSizing = "border-box"; wrap.append(text, control); panel.appendChild(wrap); return control; };
  const prompt = document.createElement("textarea"); prompt.value = get("prompt", ""); prompt.style.cssText = "width:100%;min-height:100px;box-sizing:border-box;background:#15191d;color:#eee;border:1px solid #59636e;padding:8px"; prompt.placeholder = "所有片段复用的 H3 完整提示词"; panel.appendChild(prompt);
  const lengthMode = document.createElement("select"); lengthMode.innerHTML = "<option value=\"seconds\">秒数</option><option value=\"frames\">5 帧</option>"; lengthMode.value = get("segment_length_mode", "秒数") === "5帧" ? "frames" : "seconds";
  const seconds = document.createElement("input"); seconds.type = "number"; seconds.min = "1"; seconds.max = "15"; seconds.step = "0.1"; seconds.value = get("segment_seconds", 5); seconds.style.width = "110px"; seconds.disabled = lengthMode.value === "frames"; seconds.oninput = refreshSummary; lengthMode.onchange = () => { seconds.disabled = lengthMode.value === "frames"; refreshSummary(); }; const lengthWrap = document.createElement("label"); lengthWrap.style.cssText = "display:flex;align-items:center;gap:8px;margin:8px 0"; lengthWrap.append("片段长度", lengthMode, seconds, "秒"); panel.appendChild(lengthWrap);
  const checkbox = (label, name, checked) => { const input = document.createElement("input"); input.type = "checkbox"; input.checked = !!get(name, checked); row(label, input); return input; };
  const useVideoMaterial = checkbox("将上传视频作为多模态参考素材", "use_reference_video_material", true);
  const passAudio = checkbox("传递参考视频音频", "pass_reference_video_audio", false);
  const audioCont = checkbox("开启音频上下文接续", "enable_audio_continuation", true);
  const cachePrompts = checkbox("一次性缓存全部片段的提示词向量", "cache_prompt_embeddings", true);
  const diskCachePrompts = checkbox("缓存提示词向量到硬盘（关闭一次性缓存时仅处理当前片段）", "cache_prompt_embeddings_to_disk", true);
  const keepModel = checkbox("片段间模型常驻显存（防止反复装卸）", "keep_model_loaded", true);
  const autoRun = checkbox("自动连续生成并在最后拼接", "auto_run", true);
  const skipDecode = checkbox("仅不解码 H3 音频（仍联合采样）", "skip_h3_audio_decode", false);
  const audioMode = document.createElement("select"); audioMode.innerHTML = "<option>H3 生成音频</option><option>参考视频音频</option>"; audioMode.value = get("final_audio_source", "H3 生成音频"); row("最终视频音频来源", audioMode);
  const restart = document.createElement("input"); restart.type = "text"; restart.value = get("audio_restart_segments", ""); restart.placeholder = "例如 3，6,9"; restart.style.width = "220px"; row("重新生成音频片段", restart);
  const previous = document.createElement("input"); previous.type = "text"; previous.value = get("previous_video_reference_segments", ""); previous.placeholder = "例如 2,5"; previous.style.width = "220px"; row("使用上段视频参考片段", previous);
  const videoCard = document.createElement("div"); videoCard.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;align-items:center;min-width:0;padding:8px;background:#15191d;border:1px solid #424b55;border-radius:6px;margin:12px 0"; panel.appendChild(videoCard);
  const renderVideoCard = () => {
    videoCard.replaceChildren();
    if (!video.path) { videoCard.textContent = "尚未上传参考视频"; return; }
    const thumb = document.createElement("img"); thumb.alt = video.originalName || video.name || "视频首帧"; thumb.style.cssText = "width:132px;height:76px;object-fit:contain;background:#0d1013";
    const media = document.createElement("video"); media.src = mediaUrl(video); media.preload = "metadata"; media.muted = true; media.style.display = "none";
    const capture = () => { if (!media.videoWidth || !media.videoHeight) return; const canvas = document.createElement("canvas"); canvas.width = media.videoWidth; canvas.height = media.videoHeight; const ctx = canvas.getContext("2d"); if (ctx) thumb.src = canvas.toDataURL("image/jpeg", .86); };
    media.onloadedmetadata = () => { try { media.currentTime = 0; } catch (_) {} }; media.onloadeddata = capture; media.onseeked = capture;
    const info = document.createElement("div"); info.style.cssText = "min-width:0;flex:1"; const name = document.createElement("div"); name.textContent = `视频参考：${video.originalName || video.name || "未命名"}`; name.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    const meta = document.createElement("div"); meta.style.cssText = "font-size:11px;color:#aeb7c1"; meta.textContent = `提示词标签：<Video 1>（${useVideoMaterial.checked ? "作为参考素材" : "仅用于分段与预处理"}）`;
    const insert = document.createElement("div"); insert.style.cssText = "display:flex;align-items:center;gap:5px;flex-wrap:wrap;font-size:11px;color:#c7d0da;margin-top:6px;line-height:1.5";
    const sec = document.createElement("input"); sec.type = "number"; sec.min = "0"; sec.step = ".01"; sec.value = Number(video.insert_seconds) || 0; sec.style.width = "68px"; sec.oninput = () => { video.insert_seconds = Math.max(0, Number(sec.value) || 0); };
    const fr = document.createElement("input"); fr.type = "number"; fr.min = "0"; fr.step = "1"; fr.value = Math.max(0, Math.floor(Number(video.insert_frames) || 0)); fr.style.width = "58px"; fr.oninput = () => { video.insert_frames = Math.max(0, Math.floor(Number(fr.value) || 0)); };
    const firstLabel = document.createElement("span"); firstLabel.textContent = "第一段插入："; firstLabel.style.whiteSpace = "nowrap";
    const secondsLabel = document.createElement("span"); secondsLabel.textContent = "秒 +"; secondsLabel.style.whiteSpace = "nowrap";
    const framesLabel = document.createElement("span"); framesLabel.textContent = "帧"; framesLabel.style.whiteSpace = "nowrap";
    const hint = document.createElement("span"); hint.textContent = "（均为0仅参考）"; hint.style.cssText = "white-space:nowrap;color:#aeb7c1";
    insert.append(firstLabel, sec, secondsLabel, fr, framesLabel, hint); info.append(name, meta, insert); videoCard.append(thumb, info, media);
  };
  useVideoMaterial.onchange = renderVideoCard;
  const assetList = document.createElement("div");
  // Keep enough horizontal room for the thumbnail, filename and insertion
  // controls; narrow columns make Chinese labels wrap one character per line.
  assetList.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:8px;margin:12px 0;align-items:start";
  panel.appendChild(assetList);
  const renderAssets = () => {
    assetList.replaceChildren();
    // Number labels independently by media type so mixed upload order does
    // not make the first image become <Picture 2>.
    const typeIndexes = { image: 0, audio: 0 };
    assets.forEach((asset, index) => {
      const type = asset.type === "image" ? "image" : "audio";
      const promptNumber = ++typeIndexes[type];
      const item = document.createElement("div"); item.style.cssText = `padding:8px;background:#15191d;border:1px solid #424b55;border-radius:5px;display:grid;grid-template-columns:${type === "image" ? "92px minmax(0,1fr) auto" : "minmax(0,1fr) auto"};gap:8px;align-items:start;min-width:0;box-sizing:border-box;overflow:hidden`;
      if (type === "image") {
        const thumb = document.createElement("img");
        thumb.alt = asset.originalName || asset.name || `图片${promptNumber}`;
        thumb.style.cssText = "width:92px;height:64px;object-fit:contain;background:#0d1013;flex:0 0 auto";
        if (asset.path) {
          thumb.src = mediaUrl(asset);
          thumb.onerror = () => { thumb.alt = "缩略图不可用"; };
        } else {
          thumb.alt = "等待上传图片";
        }
        item.appendChild(thumb);
      }
      const body = document.createElement("div"); body.style.cssText = "min-width:0;overflow:hidden;display:flex;flex-direction:column;gap:4px";
      const title = document.createElement("div"); title.textContent = `${type === "image" ? "图片" : "音频"}${promptNumber}：${asset.originalName || asset.name || "未命名"}`; title.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap"; body.appendChild(title);
      if (type === "image") {
        const meta = document.createElement("div"); meta.style.cssText = "font-size:11px;color:#aeb7c1"; meta.textContent = `提示词标签：<Picture ${promptNumber}> | 仅第一段插入`;
        const insert = document.createElement("div"); insert.style.cssText = "display:flex;align-items:center;gap:5px;flex-wrap:wrap;font-size:11px;color:#c7d0da;margin-top:1px;line-height:1.5";
        const sec = document.createElement("input"); sec.type = "number"; sec.min = "0"; sec.step = ".01"; sec.value = Number(asset.insert_seconds) || 0; sec.style.width = "64px"; sec.oninput = () => { asset.insert_seconds = Math.max(0, Number(sec.value) || 0); };
        const fr = document.createElement("input"); fr.type = "number"; fr.min = "0"; fr.step = "1"; fr.value = Math.max(0, Math.floor(Number(asset.insert_frames) || 0)); fr.style.width = "54px"; fr.oninput = () => { asset.insert_frames = Math.max(0, Math.floor(Number(fr.value) || 0)); };
        const firstLabel = document.createElement("span"); firstLabel.textContent = "第一段插入："; firstLabel.style.whiteSpace = "nowrap";
        const secondsLabel = document.createElement("span"); secondsLabel.textContent = "秒 +"; secondsLabel.style.whiteSpace = "nowrap";
        const framesLabel = document.createElement("span"); framesLabel.textContent = "帧"; framesLabel.style.whiteSpace = "nowrap";
        const hint = document.createElement("span"); hint.textContent = "（均为0仅作参考）"; hint.style.cssText = "white-space:nowrap;color:#aeb7c1";
        insert.append(firstLabel, sec, secondsLabel, fr, framesLabel, hint); body.appendChild(insert);
      }
      item.append(body, makeButton("删除", () => { assets.splice(index, 1); renderAssets(); })); assetList.appendChild(item);
    });
  };
  const add = async (type) => {
    try {
      let selected = [];
      const options = readPickerOptions();
      if (options.mode === "python") selected = await selectFilesWithPython(type, readDirectories()[type], options.useDefaultPath);
      else {
        const input = document.createElement("input"); input.type = "file"; input.multiple = true; input.accept = type === "image" ? "image/*" : "audio/*";
        selected = await new Promise((resolve) => { input.onchange = () => resolve(Array.from(input.files || [])); input.click(); });
      }
      const uploaded = [];
      for (const item of selected) uploaded.push(item instanceof File ? await uploadOne(item, type) : item);
      if (assets.length + uploaded.length > 11) throw new Error("附加图片与音频参考最多 11 个（另有 1 个专用参考视频）");
      assets.push(...uploaded); renderAssets(); notice.textContent = `已按选择顺序添加 ${uploaded.length} 个${type === "image" ? "图片" : "音频"}参考。`;
    } catch (error) { notice.textContent = error.message || String(error); }
  };
  const assetButtons = document.createElement("div"); assetButtons.append(makeButton("+ 添加图片", () => add("image")), makeButton("+ 添加音频", () => add("audio"))); panel.appendChild(assetButtons);
  const uploadVideo = async () => {
    try {
      let selected = [];
      const options = readPickerOptions();
      if (options.mode === "python") selected = await selectFilesWithPython("video", readDirectories().video, options.useDefaultPath);
      else { const input = document.createElement("input"); input.type = "file"; input.accept = "video/*"; selected = await new Promise((resolve) => { input.onchange = () => resolve(Array.from(input.files || []).slice(0, 1)); input.click(); }); }
      if (!selected.length) return;
      const uploaded = selected[0] instanceof File ? await uploadOne(selected[0], "video") : selected[0];
      const info = await probeVideoInfo(uploaded.path || uploaded.name);
      video = { ...uploaded, ...info, frame_count_24: info.frame_count_24 || Math.round(Number(info.duration || 0) * 24) };
      renderVideoCard();
      refreshSummary(); notice.textContent = "参考视频已上传；分段数量会按视频时长和每段秒数自动计算。";
    } catch (error) { notice.textContent = error.message || String(error); }
  };
  panel.appendChild(makeButton(video.path ? "重新上传参考视频" : "+ 上传参考视频", uploadVideo));
  const actions = document.createElement("div"); actions.style.cssText = "display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px;min-width:0;margin-top:16px";
  actions.append(makeButton("取消", () => shade.remove()), makeButton("保存", () => {
    if (!video.path) { notice.textContent = "请先上传参考视频。"; return; }
    set("prompt", prompt.value); set("segment_seconds", Number(seconds.value) || 5); set("segment_length_mode", lengthMode.value === "frames" ? "5帧" : "秒数"); set("use_reference_video_material", useVideoMaterial.checked); set("pass_reference_video_audio", passAudio.checked); set("enable_audio_continuation", audioCont.checked); set("cache_prompt_embeddings", cachePrompts.checked); set("cache_prompt_embeddings_to_disk", diskCachePrompts.checked); set("keep_model_loaded", keepModel.checked); set("auto_run", autoRun.checked); set("skip_h3_audio_decode", skipDecode.checked); set("final_audio_source", audioMode.value); set("audio_restart_segments", restart.value); set("previous_video_reference_segments", previous.value); set("reference_video_json", JSON.stringify(video, null, 2)); set("reference_assets_json", JSON.stringify(assets, null, 2)); syncSerializedWidgets(node); node.setDirtyCanvas(true, true); node.graph?.setDirtyCanvas?.(true, true); shade.remove();
  }));
  panel.appendChild(actions); shade.appendChild(panel); document.body.appendChild(shade); renderAssets(); renderVideoCard(); refreshSummary();
}


async function probeAudioDuration(path) {
  if (!path) return 0;
  try {
    const res = await fetch("/h3_auto_director/audio_info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data && Number.isFinite(data.duration) && data.duration > 0) {
        return Number(data.duration);
      }
    }
  } catch (_) {}
  return 0;
}

function getSegmentAudioAndEffectiveDuration(seg, index, cropFramesValue = 0) {
  const dur = Number(seg?.duration) || 5;
  if (index === 0) {
    return { contextFrames: 0, contextSec: 0, effectiveDuration: dur, audioDuration: dur };
  }
  const continueVideo = seg?.continue_video !== false;
  const prevVideoRef = !!seg?.use_previous_video_reference;
  if (!continueVideo || prevVideoRef) {
    return { contextFrames: 0, contextSec: 0, effectiveDuration: dur, audioDuration: dur };
  }
  const cropVal = Math.max(0, Math.floor(Number(cropFramesValue) || 0));
  const ctxFrames = cropVal > 0 ? cropVal : 22;
  const ctxSec = ctxFrames / 24.0;
  const audioDuration = Math.max(0.1, dur - ctxSec);
  return { contextFrames: ctxFrames, contextSec: ctxSec, effectiveDuration: audioDuration, audioDuration: audioDuration };
}

function adjustSegmentsToAudioDuration(segments, audioSec, cropFramesValue = 0) {
  let remainingAudio = Math.max(1.0, audioSec);
  const adjusted = [];
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (remainingAudio <= 0) break;
    const info = getSegmentAudioAndEffectiveDuration(seg, i, cropFramesValue);
    const ctxSec = info.contextSec;
    const normalAudioDur = info.audioDuration;
    if (i === segments.length - 1 || remainingAudio <= normalAudioDur) {
      seg.duration = Math.round((remainingAudio + ctxSec) * 100) / 100;
      setSegmentLengthMode(seg, "seconds");
      adjusted.push(seg);
      remainingAudio = 0;
      break;
    } else {
      adjusted.push(seg);
      remainingAudio -= normalAudioDur;
    }
  }
  if (adjusted.length > 1 && adjusted[adjusted.length - 1].duration < 1.0) {
    const last = adjusted[adjusted.length - 1];
    const prev = adjusted[adjusted.length - 2];
    const need = 1.0 - last.duration;
    if (prev.duration - need >= 1.0) {
      prev.duration = Math.round((prev.duration - need) * 100) / 100;
      last.duration = 1.0;
    }
  }
  segments.length = 0;
  segments.push(...adjusted);
}

function showAudioShortageDialog({ audioDuration, totalVideoDuration, onModifyVideoDuration, onSaveDirectly }) {
  const modalShade = document.createElement("div");
  modalShade.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:10001;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif";
  const box = document.createElement("div");
  box.style.cssText = "width:min(520px,90vw);background:#20252b;color:#eee;border:1px solid #e3b341;border-radius:8px;padding:20px;box-shadow:0 16px 50px #000;display:flex;flex-direction:column;gap:14px;box-sizing:border-box";

  const title = document.createElement("h3");
  title.textContent = "⚠️ 音频驱动时长提示";
  title.style.cssText = "margin:0;font-size:18px;color:#e3b341;display:flex;align-items:center;gap:8px";
  box.appendChild(title);

  const desc = document.createElement("div");
  desc.style.cssText = "font-size:13px;line-height:1.6;color:#c9d1d9";
  desc.innerHTML = `检测到上传的驱动音频时长（<b style="color:#58a6ff">${audioDuration.toFixed(2)} 秒</b>）短于视频计划总时长（<b style="color:#f85149">${totalVideoDuration.toFixed(2)} 秒</b>）。<br>差额：<b>${(totalVideoDuration - audioDuration).toFixed(2)} 秒</b>。<br><br>请选择处理方式：<br>• <b>修改视频时长</b>：自动缩短或调整视频片段时长以完全匹配音频时长。<br>• <b>直接保存</b>：视频总时长不变，音频不足的末端片段将直接原样音频潜空间强制替换（静音填充）。`;
  box.appendChild(desc);

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;flex-wrap:wrap;justify-content:flex-end;gap:10px;margin-top:8px";

  const cancelBtn = makeButton("取消", () => modalShade.remove());
  cancelBtn.style.cssText = "padding:6px 14px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer";

  const modifyBtn = makeButton("修改视频时长", () => {
    modalShade.remove();
    onModifyVideoDuration();
  });
  modifyBtn.style.cssText = "padding:6px 14px;background:#1f6feb;color:#fff;border:1px solid #388bfd;border-radius:6px;cursor:pointer;font-weight:600";

  const saveDirectlyBtn = makeButton("直接保存", () => {
    modalShade.remove();
    onSaveDirectly();
  });
  saveDirectlyBtn.style.cssText = "padding:6px 14px;background:#238636;color:#fff;border:1px solid #2ea043;border-radius:6px;cursor:pointer;font-weight:600";

  btnRow.append(cancelBtn, modifyBtn, saveDirectlyBtn);
  box.appendChild(btnRow);
  modalShade.appendChild(box);
  document.body.appendChild(modalShade);
}

function showContextMissingModal(data) {
  const existing = document.getElementById("h3-context-missing-dialog");
  if (existing) existing.remove();

  const modalShade = document.createElement("div");
  modalShade.id = "h3-context-missing-dialog";
  modalShade.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:10002;display:flex;align-items:center;justify-content:center;padding:16px;box-sizing:border-box;font-family:system-ui,sans-serif";

  const box = document.createElement("div");
  box.style.cssText = "width:min(640px,94vw);background:#1e242c;color:#eee;border:2px solid #e06c75;border-radius:10px;padding:22px;box-shadow:0 20px 60px rgba(0,0,0,.7);display:flex;flex-direction:column;gap:14px;box-sizing:border-box";

  const title = document.createElement("h3");
  title.innerHTML = "⚠️ H3 自动导演：上下文潜空间缺失";
  title.style.cssText = "margin:0;font-size:18px;color:#ff6b6b;display:flex;align-items:center;gap:8px;border-bottom:1px solid #3d444d;padding-bottom:10px";
  box.appendChild(title);

  const targetSeg = data.target_segment || ((data.context_segment || 0) + 1);
  const contextSeg = data.context_segment || 0;

  const desc = document.createElement("div");
  desc.style.cssText = "font-size:13px;line-height:1.6;background:#291a1d;border:1px solid #5c262a;border-radius:6px;padding:12px;color:#f8d7da";
  desc.innerHTML = `
    <div style="font-weight:bold;margin-bottom:6px;font-size:14px;color:#ff8585">❌ 报错原因分析：</div>
    <div>当前工作流正在请求生成【<b style="color:#61afef;font-size:14px">第 ${targetSeg} 段</b>】。因该段开启了视频/音频接续，必须基于上一段（<b style="color:#e5c07b;font-size:14px">第 ${contextSeg} 段</b>）生成的潜空间继续运行，但在当前工程缓存目录中<b>未找到第 ${contextSeg} 段的潜空间文件</b>。</div>
  `;
  box.appendChild(desc);

  const sol = document.createElement("div");
  sol.style.cssText = "font-size:13px;line-height:1.6;background:#171c23;border:1px solid #30363d;border-radius:6px;padding:12px;color:#c9d1d9";
  sol.innerHTML = `
    <div style="font-weight:bold;margin-bottom:8px;font-size:14px;color:#7ee787">💡 明确排查与解决方法：</div>
    <div style="margin-bottom:10px">
      <b style="color:#58a6ff">方案 1（新项目 / 重新生成整个视频）：</b><br>
      如果你想从头开始生成整个视频，请点击下方绿色按钮 <b style="color:#98c379">【↺ 一键重置片段序号为 0】</b>，即可将画布中【片段解析】节点的【上下文片段序号】重置为 0（从第 1 段开始生成）。
    </div>
    <div>
      <b style="color:#e5c07b">方案 2（断点接续生成）：</b><br>
      请检查第 <b>${contextSeg}</b> 段是否已被生成并存放在项目目录。若尚未生成，请手动将【片段解析】节点的【上下文片段序号】调整为已完成的最后一段序号。
    </div>
  `;
  box.appendChild(sol);

  const details = document.createElement("details");
  details.style.cssText = "background:#121519;border:1px solid #30363d;border-radius:6px;padding:8px 12px;font-size:12px;color:#8b949e";
  details.innerHTML = `
    <summary style="cursor:pointer;color:#aeb7c1;font-weight:bold">点击展开检查的工程路径与候选文件</summary>
    <div style="margin-top:8px;word-break:break-all;line-height:1.5">
      <div><b>项目名称：</b>${data.project_name || "未指定"}</div>
      <div><b>工程目录：</b>${data.project_dir || "未找到"}</div>
      <div style="margin-top:4px"><b>一采候选潜空间：</b>${data.stage1_path || "无"}</div>
      <div><b>二采候选潜空间：</b>${data.stage2_path || "无"}</div>
    </div>
  `;
  box.appendChild(details);

  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;flex-wrap:wrap;justify-content:flex-end;gap:10px;margin-top:4px";

  const resetBtn = makeButton("↺ 一键重置片段序号为 0 并关闭", () => {
    let resetCount = 0;
    const allNodes = app.graph?._nodes || [];
    for (const n of allNodes) {
      if (n.type === SEGMENT_NODE || n.comfyClass === SEGMENT_NODE) {
        const sw = widget(n, "segment_index");
        if (sw) {
          sw.value = 0;
          sw.callback?.(0);
          n.setDirtyCanvas?.(true, true);
          resetCount++;
        }
      }
    }
    app.graph?.setDirtyCanvas?.(true, true);
    modalShade.remove();
    alert(`已成功将 ${resetCount} 个【片段解析】节点的上下文片段序号重置为 0！\n现在点击【提示词加入队列】即可从第 1 段从头开始正常生成。`);
  });
  resetBtn.style.cssText = "padding:8px 16px;background:#238636;color:#fff;border:1px solid #2ea043;border-radius:6px;cursor:pointer;font-weight:bold;font-size:13px";

  const closeBtn = makeButton("我知道了 (关闭)", () => modalShade.remove());
  closeBtn.style.cssText = "padding:8px 16px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;cursor:pointer;font-size:13px";

  btnRow.append(resetBtn, closeBtn);
  box.appendChild(btnRow);
  modalShade.appendChild(box);
  document.body.appendChild(modalShade);
}

try {
  api.addEventListener("h3-auto-director-context-missing", (event) => {
    const data = event?.detail || {};
    showContextMissingModal(data);
  });
} catch (_) {}

function openEditor(node) {
  let segments = readSegments(node);
  if (!segments.length) segments = [{ prompt: "", duration: 5, audio_restart: false, references: [] }];
  const dirs = readDirectories();
  const pickerOptions = readPickerOptions();
  const pickerHandles = {};

  const shade = document.createElement("div");
  shade.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(1240px,96vw);height:min(88vh,900px);min-width:min(760px,calc(100vw - 24px));min-height:min(520px,calc(100vh - 24px));resize:both;overflow:auto;background:#20252b;color:#eee;border:1px solid #59636e;border-radius:8px;padding:20px;box-shadow:0 16px 60px #000;display:flex;flex-direction:column;box-sizing:border-box";
  const title = document.createElement("h2");
  title.textContent = "H3 自动导演｜片段列表";
  title.style.cssText = "margin:0 0 12px;font-size:22px";
  panel.appendChild(title);

  const dirPanel = document.createElement("div");
  dirPanel.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px;padding:10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px";
  const dirTitle = document.createElement("div");
  dirTitle.textContent = "默认打开路径只影响文件选择器，导入后的素材固定保存到 h3_refs/*。Python 模式在 ComfyUI 所在机器打开系统对话框。";
  dirTitle.style.cssText = "grid-column:1/-1;font-size:12px;color:#aeb7c1";
  dirPanel.appendChild(dirTitle);
  const pickerControls = document.createElement("div");
  pickerControls.style.cssText = "grid-column:1/-1;display:flex;flex-wrap:wrap;align-items:center;gap:14px;font-size:12px;min-width:0";
  const defaultLabel = document.createElement("label"); defaultLabel.style.cssText = "display:flex;align-items:center;gap:6px";
  const defaultToggle = document.createElement("input"); defaultToggle.type = "checkbox"; defaultToggle.checked = pickerOptions.useDefaultPath;
  defaultToggle.onchange = () => { pickerOptions.useDefaultPath = defaultToggle.checked; writePickerOptions(pickerOptions); };
  defaultLabel.append(defaultToggle, "使用默认打开路径");
  const modeLabel = document.createElement("label"); modeLabel.style.cssText = "display:flex;align-items:center;gap:6px"; modeLabel.append("文件选择方式");
  const modeSelect = document.createElement("select"); modeSelect.style.cssText = "background:#15191d;color:#eee;border:1px solid #59636e;padding:4px";
  modeSelect.innerHTML = "<option value=\"python\">Python 原生（推荐）</option><option value=\"browser\">浏览器调用</option>";
  modeSelect.value = pickerOptions.mode;
  modeSelect.onchange = () => { pickerOptions.mode = modeSelect.value; writePickerOptions(pickerOptions); };
  modeLabel.appendChild(modeSelect); pickerControls.append(defaultLabel, modeLabel); dirPanel.appendChild(pickerControls);
  ["image", "video", "audio"].forEach((type) => {
    const label = document.createElement("label");
    label.style.cssText = "display:flex;align-items:center;gap:6px;font-size:12px";
    label.append(type === "image" ? "图片" : type === "video" ? "视频" : "音频");
    const input = document.createElement("input");
    input.type = "text"; input.value = dirs[type]; input.placeholder = "例如 h3_refs/images";
    input.style.cssText = "min-width:0;flex:1;background:#15191d;color:#eee;border:1px solid #59636e;padding:5px";
    input.oninput = () => { dirs[type] = cleanSubfolder(input.value); writeDirectories(dirs); };
    const choose = makeButton("选择起始目录", async () => {
      if (!pickerOptions.useDefaultPath) { notice.textContent = "已关闭默认打开路径；开启后才能设置。"; return; }
      if (pickerOptions.mode === "python") {
        try {
          const directory = await selectDirectoryWithPython(dirs[type]);
          if (!directory) { notice.textContent = "未选择目录。"; return; }
          dirs[type] = directory.replace(/\\/g, "/"); input.value = dirs[type]; writeDirectories(dirs);
          notice.textContent = `${type === "image" ? "图片" : type === "video" ? "视频" : "音频"}默认打开目录已设置。`;
        } catch (error) { notice.textContent = `设置起始目录失败：${error.message || error}`; }
        return;
      }
      if (typeof window.showDirectoryPicker !== "function") { notice.textContent = "当前浏览器不支持设置文件选择器起始目录。"; return; }
      try {
        pickerHandles[type] = await window.showDirectoryPicker({ mode: "read" });
        dirs[type] = pickerHandles[type].name || dirs[type]; input.value = dirs[type]; writeDirectories(dirs);
        notice.textContent = `${type === "image" ? "图片" : type === "video" ? "视频" : "音频"}文件选择器起始目录已设置。`;
      } catch (error) { if (error?.name !== "AbortError") notice.textContent = `设置起始目录失败：${error.message || error}`; }
    }, "选择文件资源管理器打开目录");
    choose.style.cssText = "white-space:nowrap;padding:4px 6px";
    label.appendChild(input); label.appendChild(choose); dirPanel.appendChild(label);
  });
  panel.appendChild(dirPanel);

  const directPanel = document.createElement("details");
  directPanel.style.cssText = "margin-bottom:12px;background:#171b20;border:1px solid #424b55;border-radius:6px;padding:8px 10px";
  const directSummary = document.createElement("summary"); directSummary.textContent = "直接输入片段配置 JSON"; directSummary.style.cursor = "pointer"; directPanel.appendChild(directSummary);
  const directHelp = document.createElement("div"); directHelp.textContent = "可直接粘贴片段数组并点击应用；应用前会检查时长和素材数量。"; directHelp.style.cssText = "font-size:11px;color:#aeb7c1;margin:7px 0"; directPanel.appendChild(directHelp);
  const directInput = document.createElement("textarea"); directInput.rows = 6; directInput.value = JSON.stringify(segments, null, 2); directInput.style.cssText = "width:100%;box-sizing:border-box;resize:vertical;background:#0d1013;color:#eee;border:1px solid #59636e;padding:6px;font-family:monospace;font-size:12px"; directPanel.appendChild(directInput);
  const directActions = document.createElement("div"); directActions.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;margin-top:7px";
  directActions.appendChild(makeButton("应用 JSON", () => {
    try {
      const parsed = JSON.parse(directInput.value);
      const next = normalizeSegments(parsed);
      if (!next.length) throw new Error("片段配置必须是非空数组");
      if (!confirmSegmentReduction(segments, next.length)) { notice.textContent = "已取消减少片段。"; return; }
      if (next.some((seg) => !segmentLengthValid(seg))) throw new Error("片段时长最低 1 秒，或选择有效的 5 帧模式");
      if (next.some((seg) => totalRefs(seg) > MAX_TOTAL_REFS || Object.entries(MAX_REFS).some(([type, max]) => countRefs(seg, type) > max))) throw new Error(`每段参考素材最多 ${MAX_TOTAL_REFS} 个`);
      segments = next; directInput.value = JSON.stringify(segments, null, 2); segmentCountInput.value = segments.length; render(); notice.textContent = "已应用直接输入的片段配置。";
    } catch (error) { notice.textContent = `JSON 无效：${error.message || error}`; }
  }));
  directActions.appendChild(makeButton("仅增加/替换提示词", () => {
    try {
      const parsed = JSON.parse(directInput.value);
      const list = Array.isArray(parsed) ? parsed : (parsed && typeof parsed === "object" ? [parsed] : []);
      if (!list.length) throw new Error("输入必须是包含提示词的数组或对象");
      const incomingPrompts = list.map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") return String(item.prompt ?? item.text ?? item.positive ?? "");
        return String(item || "");
      });
      let replaced = 0;
      let added = 0;
      for (let i = 0; i < incomingPrompts.length; i++) {
        const text = incomingPrompts[i];
        if (i < segments.length) {
          segments[i].prompt = text;
          replaced++;
        } else {
          const raw = list[i];
          const seg = (raw && typeof raw === "object") ? normalizeSegment(raw) : {
            prompt: text, duration: 5, duration_mode: "seconds",
            audio_restart: false, continue_audio: true, continue_video: true,
            references: [], _references_open: false,
          };
          seg.prompt = text;
          segments.push(seg);
          added++;
        }
      }
      directInput.value = JSON.stringify(segments, null, 2);
      segmentCountInput.value = String(segments.length);
      render();
      notice.textContent = `已替换 ${replaced} 个片段的提示词（保留原有素材与时长设置）${added > 0 ? `，并追加了 ${added} 个新片段` : ""}。`;
    } catch (error) { notice.textContent = `提示词应用失败：${error.message || error}`; }
  }, "保持各分段的时长、参考素材、开关等配置不变，仅按顺序替换现有提示词；若输入提示词超出当前片段数则自动追加"));
  directActions.appendChild(makeButton("从列表更新 JSON", () => { directInput.value = JSON.stringify(segments, null, 2); }));
  directPanel.appendChild(directActions); panel.appendChild(directPanel);

  const segmentDurationPanel = document.createElement("div"); segmentDurationPanel.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:8px 10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;font-size:12px;min-width:0";
  segmentDurationPanel.append("统一片段秒数");
  const segmentSeconds = document.createElement("input"); segmentSeconds.type = "number"; segmentSeconds.min = "1"; segmentSeconds.max = "15"; segmentSeconds.step = "0.1"; segmentSeconds.value = "5"; segmentSeconds.style.cssText = "width:76px;background:#15191d;color:#eee;border:1px solid #59636e;padding:5px"; segmentDurationPanel.appendChild(segmentSeconds);
  segmentDurationPanel.appendChild(makeButton("应用到全部片段", () => {
    const value = Math.max(1, Math.min(15, Number(segmentSeconds.value) || 5));
    segmentSeconds.value = String(value);
    segments.forEach((seg) => { setSegmentLengthMode(seg, "seconds"); seg.duration = value; });
    directInput.value = JSON.stringify(segments, null, 2);
    render();
    notice.textContent = `已将全部片段的秒数设置为 ${value}。`;
  }));
  panel.appendChild(segmentDurationPanel);

  const segmentCountPanel = document.createElement("div"); segmentCountPanel.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:8px 10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;font-size:12px;min-width:0";
  segmentCountPanel.append("片段数量");
  const segmentCountInput = document.createElement("input"); segmentCountInput.type = "number"; segmentCountInput.min = "1"; segmentCountInput.max = "999"; segmentCountInput.step = "1"; segmentCountInput.value = String(segments.length); segmentCountInput.style.cssText = "width:76px;background:#15191d;color:#eee;border:1px solid #59636e;padding:5px"; segmentCountPanel.appendChild(segmentCountInput);
  segmentCountPanel.appendChild(makeButton("应用片段数量", () => {
    const target = Math.max(1, Math.min(999, Math.floor(Number(segmentCountInput.value) || segments.length)));
    if (!confirmSegmentReduction(segments, target)) { segmentCountInput.value = String(segments.length); notice.textContent = "已取消减少片段。"; return; }
    while (segments.length < target) segments.push({ prompt: "", duration: 5, duration_mode: "seconds", audio_restart: false, continue_audio: true, continue_video: true, references: [], _references_open: false });
    if (segments.length > target) segments.length = target;
    segmentCountInput.value = String(target); directInput.value = JSON.stringify(segments, null, 2); render(); notice.textContent = `片段数量已设置为 ${target}。`;
  }));
  panel.appendChild(segmentCountPanel);

  const cropPanel = document.createElement("div"); cropPanel.style.cssText = "display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:8px 10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;font-size:12px";
  cropPanel.append("自动裁剪上下文帧数");
  const cropWidget = widget(node, "auto_context_crop_frames");
  const cropFrames = document.createElement("input"); cropFrames.type = "number"; cropFrames.min = "0"; cropFrames.max = "4096"; cropFrames.step = "1";
  cropFrames.value = String(Math.max(0, Math.floor(Number(cropWidget?.value) || 0)));
  cropFrames.title = "0 表示按当前上下文长度自动计算；大于 0 时使用指定帧数";
  cropFrames.style.cssText = "width:88px;background:#15191d;color:#eee;border:1px solid #59636e;padding:5px";
  cropPanel.append(cropFrames, "帧（0=自动计算；大于 0 时自动启用裁剪）");
  cropFrames.oninput = () => { if (typeof updateAudioComparison === "function") updateAudioComparison(); };
  panel.appendChild(cropPanel);

  const deferredPanel = document.createElement("label"); deferredPanel.style.cssText = "display:flex;flex-wrap:wrap;align-items:flex-start;gap:8px;padding:8px 10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;font-size:12px;line-height:1.45;min-width:0";
  const deferredWidget = widget(node, "decode_after_all_segments");
  const deferredDecode = document.createElement("input"); deferredDecode.type = "checkbox"; deferredDecode.checked = !!deferredWidget?.value;
  deferredDecode.title = "全部采样后从硬盘逐段读取 latent、逐段解码并拼接；开启时视频上下文固定使用缓存潜空间直取。";
  deferredPanel.append(deferredDecode, "所有片段采样完成后统一解码（逐段处理，不同时占用多段显存）");
  panel.appendChild(deferredPanel);

  const skipRefPanel = document.createElement("label"); skipRefPanel.style.cssText = "display:flex;flex-wrap:wrap;align-items:flex-start;gap:8px;padding:8px 10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;font-size:12px;line-height:1.45;min-width:0";
  const skipRefWidget = widget(node, "skip_reference_encoding");
  const skipRefToggle = document.createElement("input"); skipRefToggle.type = "checkbox"; skipRefToggle.checked = !!skipRefWidget?.value;
  skipRefToggle.title = "不使用 VAE 预编码参考素材（跳过图像/视频/音频参考编码，加速且省显存）；若某素材设置了插入时间或帧，将自动强制开启素材编码以完成画面插入。";
  skipRefPanel.append(skipRefToggle, "不编码参考素材（纯文本加速；若素材设置了插入时间则自动强制编码）");
  panel.appendChild(skipRefPanel);

  const keepModelPanel = document.createElement("label"); keepModelPanel.style.cssText = "display:flex;flex-wrap:wrap;align-items:flex-start;gap:8px;padding:8px 10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;font-size:12px;line-height:1.45;min-width:0";
  const keepModelWidget = widget(node, "keep_model_loaded");
  const keepModelToggle = document.createElement("input"); keepModelToggle.type = "checkbox"; keepModelToggle.checked = keepModelWidget ? !!keepModelWidget.value : true;
  keepModelToggle.title = "在多片段连续采样过程中将扩散模型保持在显存中，避免每个片段结束时被卸载和垃圾回收导致下一片段重新加载。";
  keepModelPanel.append(keepModelToggle, "片段间模型常驻显存（防止反复装卸加速生成）");
  panel.appendChild(keepModelPanel);

  // Audio Drive Panel
  const audioDriveWidget = widget(node, "enable_audio_drive");
  const audioDriveFileWidget = widget(node, "audio_drive_file");
  let currentAudioDuration = 0;

  const audioDrivePanel = document.createElement("div");
  audioDrivePanel.style.cssText = "padding:10px 12px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;display:flex;flex-direction:column;gap:8px;font-size:12px";

  const audioDriveHeader = document.createElement("div");
  audioDriveHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap";

  const audioDriveLabel = document.createElement("label");
  audioDriveLabel.style.cssText = "display:flex;align-items:center;gap:6px;font-weight:700;cursor:pointer;color:#f0f3f6";
  const audioDriveToggle = document.createElement("input");
  audioDriveToggle.type = "checkbox";
  audioDriveToggle.checked = !!audioDriveWidget?.value;
  audioDriveLabel.append(audioDriveToggle, "启用音频驱动（切分上传音频并强制替换音频潜空间）");
  audioDriveHeader.appendChild(audioDriveLabel);
  audioDrivePanel.appendChild(audioDriveHeader);

  const audioDriveContent = document.createElement("div");
  audioDriveContent.style.cssText = `display:${audioDriveToggle.checked ? "flex" : "none"};flex-direction:column;gap:8px;margin-top:4px`;

  const audioDriveHelp = document.createElement("div");
  audioDriveHelp.style.cssText = "font-size:11px;color:#aeb7c1;line-height:1.4";
  audioDriveHelp.textContent = "开启后上传音频，插件自动根据当前片段的秒数切分音频并音频潜空间强制替换。音频时长大于视频总时长则舍弃多余音频；短于视频总时长在保存时可选择自动修改视频时长或直接保存。";
  audioDriveContent.appendChild(audioDriveHelp);

  const audioDriveFileRow = document.createElement("div");
  audioDriveFileRow.style.cssText = "display:flex;align-items:center;gap:8px;flex-wrap:wrap";

  const audioDrivePath = document.createElement("input");
  audioDrivePath.type = "text";
  audioDrivePath.value = String(audioDriveFileWidget?.value || "");
  audioDrivePath.placeholder = "上传音频后自动填写，或直接输入 input/ 相对路径 / 绝对路径";
  audioDrivePath.style.cssText = "flex:1;min-width:220px;background:#0d1013;color:#eee;border:1px solid #59636e;padding:5px;border-radius:4px";
  audioDriveFileRow.appendChild(audioDrivePath);

  const audioPlayer = document.createElement("audio");
  audioPlayer.controls = true;
  audioPlayer.preload = "metadata";
  audioPlayer.style.cssText = "width:100%;display:none;margin-top:4px";

  const audioCompareRow = document.createElement("div");
  audioCompareRow.style.cssText = "font-size:12px;color:#c9d1d9;display:flex;align-items:center;gap:12px;flex-wrap:wrap";

  const updateAudioComparison = () => {
    const cropVal = cropWidget ? Math.max(0, Math.floor(Number(cropFrames.value) || 0)) : 0;
    const segmentInfos = segments.map((s, i) => getSegmentAudioAndEffectiveDuration(s, i, cropVal));
    const totalAudioNeeded = segmentInfos.reduce((sum, item) => sum + item.audioDuration, 0);
    const rawSetDuration = segments.reduce((sum, s) => sum + (Number(s.duration) || 5), 0);

    if (!audioDriveToggle.checked || !audioDrivePath.value.trim()) {
      audioCompareRow.innerHTML = "";
      return;
    }
    const diff = totalAudioNeeded - currentAudioDuration;
    let badge = "";
    if (currentAudioDuration <= 0) {
      badge = `<span style="color:#8b949e">（等待解析音频时长…）</span>`;
    } else if (Math.abs(diff) <= 0.05) {
      badge = `<span style="color:#3fb950;font-weight:600">✓ 时长完美匹配</span>`;
    } else if (diff < 0) {
      badge = `<span style="color:#58a6ff;font-weight:600">ℹ 音频多出 ${Math.abs(diff).toFixed(2)} 秒（生成时自动舍弃多余音频）</span>`;
    } else {
      badge = `<span style="color:#e3b341;font-weight:600">⚠️ 音频短缺 ${diff.toFixed(2)} 秒（保存时将提示修改时长或直接保存）</span>`;
    }
    const hasContextReduction = Math.abs(totalAudioNeeded - rawSetDuration) > 0.01;
    const contextHint = hasContextReduction
      ? ` (接续裁剪后；设置总长 ${rawSetDuration.toFixed(2)} 秒)`
      : "";
    audioCompareRow.innerHTML = `<span>驱动音频时长：<b style="color:#58a6ff">${currentAudioDuration > 0 ? currentAudioDuration.toFixed(2) + " 秒" : "未知"}</b></span> | <span>全部片段视频总时长：<b style="color:#f0f3f6">${totalAudioNeeded.toFixed(2)} 秒</b> (${segments.length}段${contextHint})</span> | ${badge}`;
  };

  const refreshAudioPath = async (filePath) => {
    audioDrivePath.value = filePath || "";
    if (!filePath) {
      currentAudioDuration = 0;
      audioPlayer.style.display = "none";
      audioPlayer.src = "";
      updateAudioComparison();
      return;
    }
    const refObj = { type: "audio", path: filePath, name: filePath.split("/").pop() };
    audioPlayer.src = mediaUrl(refObj);
    audioPlayer.style.display = "block";
    try {
      const dur = await probeAudioDuration(filePath);
      if (dur > 0) {
        currentAudioDuration = dur;
        updateAudioComparison();
      }
    } catch (_) {}
  };

  audioPlayer.onloadedmetadata = () => {
    if (Number.isFinite(audioPlayer.duration) && audioPlayer.duration > 0) {
      currentAudioDuration = audioPlayer.duration;
      updateAudioComparison();
    }
  };

  audioDrivePath.oninput = () => {
    refreshAudioPath(audioDrivePath.value.trim());
  };

  const audioFileInput = document.createElement("input");
  audioFileInput.type = "file";
  audioFileInput.accept = "audio/*";
  audioFileInput.style.display = "none";
  audioFileInput.onchange = async () => {
    const file = audioFileInput.files?.[0];
    if (!file) return;
    try {
      notice.textContent = "正在上传驱动音频…";
      const uploaded = await uploadOne(file, "audio");
      notice.textContent = `音频上传完成：${uploaded.name}`;
      await refreshAudioPath(uploaded.path);
    } catch (err) {
      notice.textContent = `上传失败：${err.message || err}`;
    }
    audioFileInput.value = "";
  };
  audioDriveContent.appendChild(audioFileInput);

  const uploadBtn = makeButton("+ 上传音频", () => {
    audioFileInput.click();
  });
  audioDriveFileRow.appendChild(uploadBtn);

  const pythonPickBtn = makeButton("选择音频文件", async () => {
    if (pickerOptions.mode === "python") {
      try {
        notice.textContent = "正在打开 Python 系统文件选择器…";
        const selected = await selectFilesWithPython("audio", dirs["audio"], pickerOptions.useDefaultPath);
        if (selected && selected.length) {
          const item = selected[0];
          notice.textContent = `已选择音频：${item.name || item.path}`;
          await refreshAudioPath(item.path);
        } else {
          notice.textContent = "未选择文件。";
        }
      } catch (err) {
        notice.textContent = `选择文件失败：${err.message || err}`;
      }
    } else {
      audioFileInput.click();
    }
  });
  audioDriveFileRow.appendChild(pythonPickBtn);

  audioDriveContent.appendChild(audioDriveFileRow);
  audioDriveContent.appendChild(audioPlayer);
  audioDriveContent.appendChild(audioCompareRow);
  audioDrivePanel.appendChild(audioDriveContent);
  panel.appendChild(audioDrivePanel);

  if (audioDrivePath.value.trim()) {
    refreshAudioPath(audioDrivePath.value.trim());
  }

  audioDriveToggle.onchange = () => {
    audioDriveContent.style.display = audioDriveToggle.checked ? "flex" : "none";
    updateAudioComparison();
  };

  const list = document.createElement("div");
  list.style.cssText = "flex:1;overflow:auto;padding-right:4px";
  panel.appendChild(list);
  const notice = document.createElement("div");
  notice.style.cssText = "min-height:20px;margin-top:6px;color:#ffbf69;font-size:12px";
  panel.appendChild(notice);

  const referencePromptNumber = (seg, ref, type) => {
    const refs = (seg.references || []).filter((item) => item.type === type);
    const index = refs.indexOf(ref);
    return type === "audio" ? videoAudioRefs(seg).length + index + 1 : index + 1;
  };

  const videoAudioPromptNumber = (seg, ref) => {
    const videos = (seg.references || []).filter((item) => item.type === "video");
    return videos.slice(0, videos.indexOf(ref) + 1).filter((item) => item.has_audio !== false && item.video_audio_enabled !== false).length;
  };

  const createRefCard = (seg, ref, type, renderRefs) => {
    const promptNumber = referencePromptNumber(seg, ref, type);
    const promptTag = `<${type === "image" ? "Picture" : type === "video" ? "Video" : "Audio"} ${promptNumber}>`;
    const mediaDetails = () => {
      const position = Number(ref.insert_seconds) || Number(ref.insert_frames) ? ` | 插入：${Number(ref.insert_seconds) || 0} 秒 + ${Number(ref.insert_frames) || 0} 帧` : " | 仅多模态参考";
      return `提示词标签：${promptTag} | 时长：${formatDuration(ref.duration)}${position}`;
    };
    const card = document.createElement("div");
    card.style.cssText = `position:relative;display:grid;grid-template-columns:${type === "audio" ? "1fr 30px" : "110px 1fr 30px"};gap:8px;align-items:center;padding:6px;background:#15191d;border:1px solid #424b55;border-radius:5px`;
    const preview = document.createElement("div");
    preview.style.cssText = "position:relative;width:104px;height:72px;background:#0d1013;border:1px dashed #59636e;display:flex;align-items:center;justify-content:center;overflow:hidden;color:#8e99a5;font-size:11px;text-align:center";
    const badge = document.createElement("span");
    badge.textContent = `${type === "image" ? "图片" : type === "video" ? "视频" : "音频"}${promptNumber}`;
    badge.style.cssText = "position:absolute;top:3px;right:3px;color:#fff;font-size:11px;font-weight:700;-webkit-text-stroke:2px #000;text-shadow:0 1px 2px #000;paint-order:stroke fill;z-index:2";
    preview.appendChild(badge);
    let mediaMeta = null;
    let audioControl = null;
    if (ref.path) {
      if (type === "image") {
        const image = document.createElement("img"); image.src = mediaUrl(ref); image.alt = ref.originalName || ref.name; image.style.cssText = "width:100%;height:100%;object-fit:contain"; preview.appendChild(image);
      } else if (type === "video") {
        if (ref.has_audio === undefined && !VIDEO_AUDIO_PROBES.has(ref)) {
          VIDEO_AUDIO_PROBES.add(ref);
          probeVideoAudio(ref.path).then((hasAudio) => {
            ref.has_audio = hasAudio;
            if (ref.video_audio_enabled === undefined) ref.video_audio_enabled = true;
            renderRefs();
          }).catch(() => { ref.has_audio = null; renderRefs(); });
        }
        // Use a canvas-extracted first frame as the stable thumbnail. Keeping
        // the video element hidden avoids animated previews and layout jumps.
        const thumbnail = document.createElement("img");
        thumbnail.alt = ref.originalName || ref.name || "视频首帧";
        thumbnail.style.cssText = "width:100%;height:100%;object-fit:contain;display:none";
        const video = document.createElement("video");
        video.src = mediaUrl(ref); video.preload = "metadata"; video.muted = true; video.playsInline = true; video.style.display = "none";
        const captureFirstFrame = () => {
          if (!video.videoWidth || !video.videoHeight) return;
          const canvas = document.createElement("canvas"); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
          const context = canvas.getContext("2d"); if (!context) return;
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          thumbnail.src = canvas.toDataURL("image/jpeg", 0.86); thumbnail.style.display = "block";
        };
        video.onloadedmetadata = () => {
          if (Number.isFinite(video.duration)) {
            ref.duration = video.duration;
            if (mediaMeta) mediaMeta.textContent = mediaDetails();
          }
          try { video.currentTime = 0; } catch (_) { /* metadata may not be seekable yet */ }
        };
        video.onloadeddata = captureFirstFrame;
        video.onseeked = captureFirstFrame;
        video.onerror = () => { thumbnail.style.display = "none"; video.style.display = "block"; video.controls = true; };
        preview.append(thumbnail, video);
      } else {
        // Audio has no visual thumbnail; keep only its player in the details column.
        audioControl = document.createElement("audio"); audioControl.src = mediaUrl(ref); audioControl.controls = true; audioControl.preload = "metadata"; audioControl.style.cssText = "width:100%";
        audioControl.onloadedmetadata = () => { if (Number.isFinite(audioControl.duration)) { ref.duration = audioControl.duration; if (mediaMeta) mediaMeta.textContent = mediaDetails(); } };
      }
    } else if (type !== "audio") preview.append("等待上传");
    if (type !== "audio") card.appendChild(preview);
    const info = document.createElement("div");
    info.style.cssText = "min-width:0;display:flex;flex-direction:column;gap:5px";
    if (audioControl) info.appendChild(audioControl);
    const name = document.createElement("div"); name.textContent = ref.originalName || ref.name || "未命名素材"; name.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap"; info.appendChild(name);
    const meta = document.createElement("div"); meta.textContent = type === "image" ? `提示词标签：${promptTag}` : mediaDetails(); meta.style.cssText = "font-size:11px;color:#aeb7c1"; mediaMeta = meta; info.appendChild(meta);
    const path = document.createElement("input"); path.type = "text"; path.value = ref.path || ""; path.placeholder = "上传后自动填写，也可手动输入 input 下路径"; path.style.cssText = "width:100%;box-sizing:border-box;background:#0d1013;color:#ddd;border:1px solid #424b55;padding:4px"; path.oninput = () => { ref.path = path.value; ref.name = path.value.split("/").pop(); renderRefs(); }; info.appendChild(path);
    const insert = document.createElement("div"); insert.style.cssText = "display:flex;align-items:center;gap:5px;flex-wrap:wrap;font-size:11px;color:#c7d0da";
    const insertSeconds = document.createElement("input"); insertSeconds.type = "number"; insertSeconds.min = "0"; insertSeconds.step = "0.01"; insertSeconds.value = Number(ref.insert_seconds) || 0; insertSeconds.style.cssText = "width:72px;background:#0d1013;color:#ddd;border:1px solid #424b55;padding:3px"; insertSeconds.oninput = () => { ref.insert_seconds = Math.max(0, Number(insertSeconds.value) || 0); mediaMeta.textContent = type === "image" ? `提示词标签：${promptTag} | 插入：${ref.insert_seconds} 秒 + ${Number(ref.insert_frames) || 0} 帧` : mediaDetails(); };
    const insertFrames = document.createElement("input"); insertFrames.type = "number"; insertFrames.min = "0"; insertFrames.step = "1"; insertFrames.value = Math.max(0, Math.floor(Number(ref.insert_frames) || 0)); insertFrames.style.cssText = "width:62px;background:#0d1013;color:#ddd;border:1px solid #424b55;padding:3px"; insertFrames.oninput = () => { ref.insert_frames = Math.max(0, Math.floor(Number(insertFrames.value) || 0)); mediaMeta.textContent = type === "image" ? `提示词标签：${promptTag} | 插入：${Number(ref.insert_seconds) || 0} 秒 + ${ref.insert_frames} 帧` : mediaDetails(); };
    insert.append("插入时间", insertSeconds, "秒 +", insertFrames, "帧（均为 0 时仅作参考）"); info.appendChild(insert);
    card.appendChild(info);
    card.appendChild(makeButton("×", () => { seg.references.splice(seg.references.indexOf(ref), 1); renderRefs(); }, "删除素材"));
    return card;
  };

  const createVideoAudioCard = (seg, videoRef, renderRefs) => {
    const number = videoAudioPromptNumber(seg, videoRef);
    const card = document.createElement("div");
    // The toggle contains both a checkbox and a label. A fixed 30px column
    // squeezed the label outside the card on narrow plan panels.
    card.style.cssText = "width:100%;box-sizing:border-box;display:grid;grid-template-columns:minmax(0,1fr) minmax(104px,max-content);gap:8px;align-items:center;padding:8px;background:#171b20;border:1px solid #424b55;border-radius:5px;overflow:hidden";
    const info = document.createElement("div");
    info.style.cssText = "min-width:0;display:flex;flex-direction:column;gap:5px";
    const title = document.createElement("div");
    title.textContent = number > 0 ? `视频音频 ${number}` : "视频音频（未检测到音轨）";
    title.style.cssText = "font-weight:700;color:#f0f3f6";
    info.appendChild(title);
    const meta = document.createElement("div");
    meta.textContent = number > 0
      ? `提示词标签：<Audio ${number}> | 来源：${videoRef.originalName || videoRef.name || "视频参考"}`
      : `未检测到音轨；仍可保留开关设置 | 来源：${videoRef.originalName || videoRef.name || "视频参考"}`;
    meta.style.cssText = "font-size:11px;color:#aeb7c1";
    info.appendChild(meta);
    const toggle = document.createElement("label");
    toggle.style.cssText = "min-width:0;display:flex;align-items:center;justify-content:flex-start;gap:6px;font-size:12px;line-height:1.35;color:#d6dde5;white-space:normal;overflow-wrap:anywhere;cursor:pointer";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.style.cssText = "flex:0 0 auto;margin:0";
    checkbox.checked = videoRef.video_audio_enabled !== false;
    checkbox.title = "关闭后只传递视频画面";
    checkbox.onchange = () => { videoRef.video_audio_enabled = checkbox.checked; renderRefs(); };
    const toggleText = document.createElement("span");
    toggleText.textContent = "传递音频";
    toggleText.style.cssText = "min-width:0;overflow-wrap:anywhere";
    toggle.append(checkbox, toggleText);
    card.append(info, toggle);
    return card;
  };

  const render = () => {
    segmentCountInput.value = String(segments.length);
    list.innerHTML = "";
    segments.forEach((seg, index) => {
      const row = document.createElement("div");
      row.style.cssText = "margin:8px 0;padding:10px;background:#2b3138;border:1px solid #424b55;border-radius:6px;box-sizing:border-box;overflow:hidden;min-width:0";
      // Keep the prompt on its own row. Putting it beside the long Chinese
      // switches made the old fixed grid squeeze labels into vertical text on
      // narrow panels and on high-DPI displays.
      const head = document.createElement("div"); head.style.cssText = "display:flex;flex-direction:column;gap:8px;width:100%;min-width:0";
      const promptRow = document.createElement("div"); promptRow.style.cssText = "display:flex;gap:8px;align-items:flex-start;width:100%;min-width:0";
      const label = document.createElement("strong"); label.textContent = String(index + 1); label.style.cssText = "flex:0 0 28px;padding-top:7px"; promptRow.appendChild(label);
      const prompt = document.createElement("textarea"); prompt.rows = 3; prompt.placeholder = "输入本片段提示词，使用 <Picture 1>/<Video 1>/<Audio 1> 标签"; prompt.value = seg.prompt; prompt.style.cssText = "flex:1 1 auto;min-width:0;width:100%;resize:vertical;background:#15191d;color:#eee;border:1px solid #59636e;padding:6px;box-sizing:border-box"; prompt.oninput = () => { seg.prompt = prompt.value; }; promptRow.appendChild(prompt); head.appendChild(promptRow);
      const controls = document.createElement("div"); controls.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start;width:100%;min-width:0;padding-left:36px;box-sizing:border-box";
      const mode = document.createElement("select"); mode.innerHTML = "<option value=\"seconds\">秒数</option><option value=\"frames\">5 帧</option>"; mode.value = seg.duration_mode || "seconds"; mode.style.cssText = "flex:0 0 72px;width:72px;background:#15191d;color:#eee;border:1px solid #59636e;padding:6px"; mode.onchange = () => { setSegmentLengthMode(seg, mode.value); render(); }; controls.appendChild(mode);
      const duration = document.createElement("input"); duration.type = "number"; duration.min = "1"; duration.max = "15"; duration.step = "0.1"; duration.value = Number(seg.duration) >= 1 ? seg.duration : 5; duration.title = "片段时长（秒）；也可选择 5 帧"; duration.disabled = mode.value === "frames"; duration.style.cssText = "flex:0 0 84px;width:84px;background:#15191d;color:#eee;border:1px solid #59636e;padding:6px;box-sizing:border-box"; duration.oninput = () => { seg.duration = Math.max(1, Number(duration.value) || 5); }; controls.appendChild(duration);
      const reset = document.createElement("label"); reset.style.cssText = "flex:0 1 220px;min-width:170px;max-width:320px;font-size:12px;display:flex;gap:5px;align-items:flex-start;padding-top:7px;line-height:1.4;white-space:normal;overflow-wrap:anywhere;word-break:break-word"; const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.style.cssText = "flex:0 0 auto;margin-top:2px"; checkbox.checked = seg.audio_restart; checkbox.onchange = () => { seg.audio_restart = checkbox.checked; }; const resetText = document.createElement("span"); resetText.textContent = "重新生成音频"; reset.append(checkbox, resetText); controls.appendChild(reset);
      const videoReset = document.createElement("label"); videoReset.style.cssText = "flex:0 1 260px;min-width:190px;max-width:360px;font-size:12px;display:flex;gap:5px;align-items:flex-start;padding-top:7px;line-height:1.4;white-space:normal;overflow-wrap:anywhere;word-break:break-word";
      const videoResetCheckbox = document.createElement("input"); videoResetCheckbox.type = "checkbox"; videoResetCheckbox.checked = index > 0 && seg.continue_video === false; videoResetCheckbox.disabled = index === 0;
      videoResetCheckbox.onchange = () => { seg.continue_video = !videoResetCheckbox.checked; if (videoResetCheckbox.checked) { previousCheckbox.checked = false; seg.use_previous_video_reference = false; } };
      const videoResetText = document.createElement("span"); videoResetText.textContent = "重新生成视频（关闭视频上下文）"; videoReset.append(videoResetCheckbox, videoResetText); controls.appendChild(videoReset);
      const previous = document.createElement("label"); previous.style.cssText = "flex:0 1 300px;min-width:210px;max-width:400px;font-size:12px;display:flex;gap:5px;align-items:flex-start;padding-top:7px;line-height:1.4;white-space:normal;overflow-wrap:anywhere;word-break:break-word";
      const previousCheckbox = document.createElement("input"); previousCheckbox.type = "checkbox"; previousCheckbox.checked = index > 0 && !!seg.use_previous_video_reference; previousCheckbox.disabled = index === 0;
      previousCheckbox.onchange = () => { seg.use_previous_video_reference = previousCheckbox.checked; if (previousCheckbox.checked) { seg.continue_video = false; videoResetCheckbox.checked = true; } };
      const previousText = document.createElement("span"); previousText.textContent = "使用上片段视频作为参考素材接续"; previous.append(previousCheckbox, previousText); controls.appendChild(previous);
      controls.appendChild(makeButton("×", () => {
        if (segments.length <= 1) return;
        if (!confirmSegmentReduction(segments, segments.length - 1, [seg])) { notice.textContent = "已取消删除片段。"; return; }
        segments.splice(index, 1); render();
      }, "删除片段")); head.appendChild(controls); row.appendChild(head);

      const details = document.createElement("details"); details.open = seg._references_open === true || (seg._references_open === undefined && (seg.references || []).length > 0); details.ontoggle = () => { seg._references_open = details.open; }; details.style.cssText = "margin-top:8px;background:#20252b;border:1px solid #59636e;border-radius:5px;padding:6px 8px";
      const summary = document.createElement("summary"); summary.textContent = `多模态参考素材（${totalRefs(seg)}/${MAX_TOTAL_REFS} / 图片${countRefs(seg, "image")}/9，视频${countRefs(seg, "video")}/3，音频${countRefs(seg, "audio")}/3）`; summary.style.cursor = "pointer"; details.appendChild(summary);
      const refList = document.createElement("div"); refList.style.cssText = "display:flex;flex-direction:column;gap:8px;margin-top:8px";
      const renderRefs = () => {
        refList.innerHTML = "";
        ["image", "video", "audio"].forEach((type) => {
          const refs = (seg.references || []).filter((ref) => ref.type === type);
          if (!refs.length) return;
          const group = document.createElement("div"); group.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:7px";
          const heading = document.createElement("div"); heading.textContent = type === "image" ? "图片参考（提示词标签从 <Picture 1> 起始编号）" : type === "video" ? "视频参考（提示词标签从 <Video 1> 起始编号）" : "音频参考（按上传顺序使用 <Audio 1>、<Audio 2>…）"; heading.style.cssText = "grid-column:1/-1;color:#c7d0da;font-size:12px"; group.appendChild(heading);
          refs.forEach((ref) => {
            group.appendChild(createRefCard(seg, ref, type, () => { summary.textContent = `多模态参考素材（${totalRefs(seg)}/${MAX_TOTAL_REFS} / 图片${countRefs(seg, "image")}/9，视频${countRefs(seg, "video")}/3，音频${countRefs(seg, "audio")}/3）`; renderRefs(); }));
            if (type === "video") group.appendChild(createVideoAudioCard(seg, ref, renderRefs));
          });
          refList.appendChild(group);
        });
      };
      const addFiles = async (type, files) => {
        const selected = Array.from(files || []);
        const remaining = MAX_REFS[type] - countRefs(seg, type);
        const totalRemaining = MAX_TOTAL_REFS - totalRefs(seg);
        if (type === "video" && seg.use_previous_video_reference && index > 0
            && countRefs(seg, "video") + selected.length > 1) {
          notice.textContent = `第 ${index + 1} 段启用上片段视频参考时，最多只能上传 1 个视频参考素材。`;
          return;
        }
        if (selected.length > totalRemaining) { notice.textContent = `第 ${index + 1} 段参考素材总数最多 ${MAX_TOTAL_REFS} 个，已拒绝本次超出上限的文件。`; return; }
        if (selected.length > remaining) { notice.textContent = `第 ${index + 1} 段的${type === "image" ? "图片" : type === "video" ? "视频" : "音频"}最多 ${MAX_REFS[type]} 个，已拒绝超出上限的文件。`; return; }
        try {
          for (const file of selected) seg.references.push(await uploadOne(file, type));
          notice.textContent = selected.length ? `第 ${index + 1} 段已按选择顺序上传 ${selected.length} 个${type === "image" ? "图片" : type === "video" ? "视频" : "音频"}。` : "";
          render();
        } catch (error) { notice.textContent = error.message || String(error); }
      };
      const addPythonFiles = async (type) => {
        try {
          notice.textContent = "正在打开 Python 系统文件选择器…";
          const selected = await selectFilesWithPython(type, dirs[type], pickerOptions.useDefaultPath);
          const remaining = MAX_REFS[type] - countRefs(seg, type);
          const totalRemaining = MAX_TOTAL_REFS - totalRefs(seg);
          if (type === "video" && seg.use_previous_video_reference && index > 0
              && countRefs(seg, "video") + selected.length > 1) {
            throw new Error(`第 ${index + 1} 段启用上片段视频参考时，最多只能上传 1 个视频参考素材。`);
          }
          if (selected.length > totalRemaining) throw new Error(`第 ${index + 1} 段参考素材总数最多 ${MAX_TOTAL_REFS} 个，已拒绝本次选择。`);
          if (selected.length > remaining) throw new Error(`第 ${index + 1} 段的${type === "image" ? "图片" : type === "video" ? "视频" : "音频"}最多 ${MAX_REFS[type]} 个，已拒绝本次选择。`);
          seg.references.push(...selected);
          notice.textContent = selected.length ? `第 ${index + 1} 段已按选择顺序导入 ${selected.length} 个${type === "image" ? "图片" : type === "video" ? "视频" : "音频"}。` : "未选择文件。";
          render();
        } catch (error) { notice.textContent = error.message || String(error); }
      };
      ["image", "video", "audio"].forEach((type) => {
        const input = document.createElement("input"); input.type = "file"; input.multiple = true; input.accept = type === "image" ? "image/*" : type === "video" ? "video/*" : "audio/*"; input.style.display = "none"; input.onchange = () => { addFiles(type, input.files); input.value = ""; }; details.appendChild(input);
        const openPicker = async () => {
          if (pickerOptions.useDefaultPath && pickerHandles[type] && typeof window.showOpenFilePicker === "function") {
            try {
              const kinds = type === "image" ? [{ description: "图片", accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp", ".bmp"] } }] : type === "video" ? [{ description: "视频", accept: { "video/*": [".mp4", ".mov", ".webm", ".mkv"] } }] : [{ description: "音频", accept: { "audio/*": [".wav", ".mp3", ".flac", ".ogg", ".m4a"] } }];
              const handles = await window.showOpenFilePicker({ multiple: true, startIn: pickerHandles[type], types: kinds, excludeAcceptAllOption: false });
              await addFiles(type, await Promise.all(handles.map((handle) => handle.getFile())));
              return;
            } catch (error) { if (error?.name === "AbortError") return; notice.textContent = `打开文件选择器失败：${error.message || error}`; }
          }
          input.click();
        };
        const addReference = () => pickerOptions.mode === "python" ? addPythonFiles(type) : openPicker();
        const button = makeButton(type === "image" ? "+ 添加图片" : type === "video" ? "+ 添加视频" : "+ 添加音频", addReference, "使用上方选择的文件选择方式"); button.style.marginRight = "6px"; details.appendChild(button);
      });
      renderRefs(); details.appendChild(refList); row.appendChild(details); list.appendChild(row);
    });
    updateAudioComparison();
  };
  render();

  const actions = document.createElement("div"); actions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end;align-items:center;margin-top:12px;min-width:0";
  const resetSegLabel = document.createElement("label");
  resetSegLabel.style.cssText = "display:inline-flex;align-items:center;gap:6px;font-size:12px;color:#cbd5e1;cursor:pointer;margin-right:auto;user-select:none";
  const resetSegCheckbox = document.createElement("input");
  resetSegCheckbox.type = "checkbox";
  resetSegCheckbox.checked = true;
  resetSegLabel.appendChild(resetSegCheckbox);
  resetSegLabel.append("保存时重置片段序号为 0（从第 1 段从头开始生成）");
  actions.appendChild(resetSegLabel);

  actions.appendChild(makeButton("+ 添加片段", () => { segments.push({ prompt: "", duration: 5, audio_restart: false, continue_video: true, use_previous_video_reference: false, references: [] }); render(); }));
  actions.appendChild(makeButton("将第 1 段参考素材应用到全部", () => { const refs = JSON.parse(JSON.stringify(segments[0].references || [])); segments.forEach((seg) => { seg.references = JSON.parse(JSON.stringify(refs)); }); render(); }));
  actions.appendChild(makeButton("取消", () => shade.remove()));
  actions.appendChild(makeButton("保存", () => {
    const invalid = segments.findIndex((seg) => totalRefs(seg) > MAX_TOTAL_REFS || Object.entries(MAX_REFS).some(([type, max]) => countRefs(seg, type) > max));
    if (invalid >= 0) { notice.textContent = `第 ${invalid + 1} 段参考素材数量超过限制（每段最多 ${MAX_TOTAL_REFS} 个）。`; return; }
    const videoInvalid = segments.findIndex((seg, index) => index > 0 && seg.use_previous_video_reference && countRefs(seg, "video") > 1);
    if (videoInvalid >= 0) {
      notice.textContent = `第 ${videoInvalid + 1} 段启用上片段视频参考时，最多只能上传 1 个视频参考素材。`;
      return;
    }
    const insertInvalid = segments.findIndex((seg) => (seg.references || []).some((ref) => Number(ref.insert_seconds) < 0 || Number(ref.insert_frames) < 0));
    if (insertInvalid >= 0) { notice.textContent = `第 ${insertInvalid + 1} 段存在无效的素材插入时间。`; return; }

    const performSave = () => {
      writeSegments(node, segments);
      if (cropWidget) {
        cropWidget.value = Math.max(0, Math.min(4096, Math.floor(Number(cropFrames.value) || 0)));
        cropWidget.callback?.(cropWidget.value);
      }
      if (deferredWidget) {
        deferredWidget.value = deferredDecode.checked;
        deferredWidget.callback?.(deferredDecode.checked);
      }
      if (skipRefWidget) {
        skipRefWidget.value = skipRefToggle.checked;
        skipRefWidget.callback?.(skipRefToggle.checked);
      }
      if (keepModelWidget) {
        keepModelWidget.value = keepModelToggle.checked;
        keepModelWidget.callback?.(keepModelToggle.checked);
      }
      if (audioDriveWidget) {
        audioDriveWidget.value = audioDriveToggle.checked;
        audioDriveWidget.callback?.(audioDriveWidget.value);
      }
      if (audioDriveFileWidget) {
        audioDriveFileWidget.value = audioDrivePath.value.trim();
        audioDriveFileWidget.callback?.(audioDriveFileWidget.value);
      }
      if (resetSegCheckbox.checked) {
        const segNodes = (node.graph?._nodes || app.graph?._nodes || []).filter((n) => n.type === SEGMENT_NODE || n.comfyClass === SEGMENT_NODE);
        for (const sn of segNodes) {
          const sw = widget(sn, "segment_index");
          if (sw && sw.value !== 0) {
            sw.value = 0;
            sw.callback?.(0);
            sn.setDirtyCanvas?.(true, true);
          }
        }
      }
      syncSerializedWidgets(node);
      node.setDirtyCanvas?.(true, true);
      node.graph?.setDirtyCanvas?.(true, true);
      shade.remove();
    };

    if (audioDriveToggle.checked && audioDrivePath.value.trim()) {
      const cropVal = Math.max(0, Math.floor(Number(cropFrames.value) || 0));
      const totalAudioNeeded = segments.reduce((sum, seg, i) => sum + getSegmentAudioAndEffectiveDuration(seg, i, cropVal).audioDuration, 0);
      const audioSec = currentAudioDuration;
      if (audioSec > 0 && audioSec < totalAudioNeeded - 0.05) {
        showAudioShortageDialog({
          audioDuration: audioSec,
          totalVideoDuration: totalAudioNeeded,
          onModifyVideoDuration: () => {
            adjustSegmentsToAudioDuration(segments, audioSec, cropVal);
            directInput.value = JSON.stringify(segments, null, 2);
            segmentCountInput.value = String(segments.length);
            render();
            notice.textContent = `已自动将视频总时长调整为 ${audioSec.toFixed(2)} 秒（已适配接续上下文扣除）。请确认无误后再次点击保存。`;
          },
          onSaveDirectly: () => {
            performSave();
          },
        });
        return;
      }
    }

    performSave();
  }));
  panel.appendChild(actions); shade.appendChild(panel); document.body.appendChild(shade);
}

app.registerExtension({
  name: "H3AutoDirector.Editor",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        const legacy = Array.isArray(info?.inputs)
          && info.inputs.some((input) => input?.name === "use_previous_video_reference");
        if (legacy && Array.isArray(info.widgets_values) && info.widgets_values.length >= 10) {
          info = { ...info, widgets_values: info.widgets_values.slice(0, 6).concat(info.widgets_values.slice(7)) };
        }
        if (Array.isArray(info?.widgets_values)) {
          let values = [...info.widgets_values];
          if (typeof values[9] === "boolean" || values.length <= 14) {
            values.splice(9, 0, "", false);
          }
          if (typeof values[9] === "boolean" || values[9] === "true" || values[9] === "false") {
            values[9] = "";
          }
          if (values[10] === "覆盖已有文件" || values[10] === "true") {
            values[10] = false;
          }
          if (values.length > 13 && (Number.isNaN(Number(values[13])) || typeof values[13] !== "number")) {
            values[13] = 0;
          }
          info = { ...info, widgets_values: values };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === MOTION_CONTEXT_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        let inputs = Array.isArray(info?.inputs) ? info.inputs : [];
        const noiseNames = new Set(["context_noise_schedule", "context_noise_seed", "context_noise_mode", "context_noise_sample_strength", "context_noise_strength"]);
        let values = Array.isArray(info?.widgets_values) ? [...info.widgets_values] : info?.widgets_values;
        if (inputs.some((input) => noiseNames.has(input?.name))) {
          inputs = inputs.filter((input) => !noiseNames.has(input?.name));
          values = values ? values.slice(0, 5) : values;
        }
        const hasMethod = inputs.some((input) => input?.name === "context_method");
        const redrawInputs = [
          ["context_sampled_start_tokens", "首部可采样 latent token 数", "INT"],
          ["context_sampled_start_strength", "首部 token 重绘强度", "FLOAT"],
          ["context_sampled_tokens", "末端可采样 latent token 数", "INT"],
          ["context_sampled_strength", "末端 token 重绘强度", "FLOAT"],
        ];
        if (hasMethod) {
          const missing = redrawInputs.filter(([name]) => !inputs.some((input) => input?.name === name));
          if (missing.length) {
            inputs = inputs.concat(missing.map(([name, label, type]) => ({ label, localized_name: label, name, type, widget: { name } })));
            if (values) values = values.concat([0, 0.25, 2, 0.25].slice(0, missing.length));
          }
        }
        if (Array.isArray(values)) {
          for (let i = 0; i < values.length; i++) {
            if (values[i] === "缓存视频 latent 直取" || values[i] === "自动（latent 优先）" || values[i] === "帧 Guide VAE 回退") {
              values[i] = "潜空间直取";
            }
          }
        }
        if (inputs !== info?.inputs || values !== info?.widgets_values) {
          info = { ...info, inputs, widgets_values: values };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === CONTROLLER_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        let inputs = Array.isArray(info?.inputs) ? info.inputs : [];
        let values = Array.isArray(info?.widgets_values) ? [...info.widgets_values] : info?.widgets_values;
        const cropIndex = inputs.findIndex((input) => input?.name === "crop_context_on_assemble");
        if (cropIndex >= 0) {
          if (values && values.length > 0) values.pop();
          inputs = inputs.filter((input) => input?.name !== "crop_context_on_assemble");
        }
        if (values && values.length > 0) {
          const formats = new Set(["mp4", "mkv", "mov", "webm"]);
          const fmtIdx = values.findIndex((v) => typeof v === "string" && formats.has(v.toLowerCase()));
          if (fmtIdx > 0 && typeof values[fmtIdx - 1] === "string" && !formats.has(values[fmtIdx - 1].toLowerCase())) {
            values.splice(fmtIdx - 1, 1);
          }
        }
        info = { ...info, inputs: inputs.filter((input) => input?.name !== "output_root"), widgets_values: values };
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === SAVE_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        let inputs = Array.isArray(info?.inputs) ? info.inputs : [];
        let values = Array.isArray(info?.widgets_values) ? [...info.widgets_values] : info?.widgets_values;
        if (values && values.length > 0) {
          const formats = new Set(["mp4", "mkv", "mov", "webm"]);
          const fmtIdx = values.findIndex((v) => typeof v === "string" && formats.has(v.toLowerCase()));
          if (fmtIdx > 0 && typeof values[fmtIdx - 1] === "string" && !formats.has(values[fmtIdx - 1].toLowerCase())) {
            values.splice(fmtIdx - 1, 1);
          }
        }
        info = { ...info, inputs: inputs.filter((input) => input?.name !== "output_root"), widgets_values: values };
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === CONTEXT_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        let inputs = Array.isArray(info?.inputs) ? info.inputs : [];
        let values = Array.isArray(info?.widgets_values) ? [...info.widgets_values] : info?.widgets_values;
        inputs = inputs.filter((input) => input?.name !== "context_stage");
        if (values && values.length > 1) {
          values = values.slice(0, 1);
        }
        info = { ...info, inputs, widgets_values: values };
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === TTS_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        const inputs = Array.isArray(info?.inputs) ? info.inputs : [];
        const names = inputs.map((input) => input?.name);
        const legacyNames = ["reference_video_json", "reference_assets_json", "pass_reference_video_audio", "audio_restart_segments"];
        if (legacyNames.some((name) => names.includes(name)) && Array.isArray(info.widgets_values)) {
          const keep = ["project_id", "segments_json", "auto_run", "cache_prompt_embeddings", "enable_audio_continuation", "concat_final_audio", "output_root", "global_reference_set"];
          const values = Object.fromEntries(names.map((name, index) => [name, info.widgets_values[index]]));
          info = { ...info, widgets_values: keep.map((name) => values[name]) };
        }
        if (Array.isArray(info?.widgets_values)) {
          let values = [...info.widgets_values];
          if (typeof values[7] === "boolean" || values.length <= 9) {
            values.splice(7, 0, "", false);
          }
          info = { ...info, widgets_values: values };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === TRANSFER_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        const names = Array.isArray(info?.inputs) ? info.inputs : [];
        const retired = ["freeze_video_sampling", "freeze_audio_sampling"];
        let values = Array.isArray(info?.widgets_values) ? [...info.widgets_values] : info?.widgets_values;
        if (retired.some((name) => names.includes(name)) && Array.isArray(values)) {
          values = values.filter((_, index) => !retired.includes(names[index]));
        }
        if (Array.isArray(values)) {
          if (typeof values[14] === "boolean" || values.length <= 16) {
            values.splice(14, 0, "", false);
          }
        }
        info = { ...info, widgets_values: values };
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === CONTROL_PREPROCESS_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        // Migrate the former single-control schema (control_type + style) to
        // the unified pose/depth node without retaining the retired style UI.
        const inputs = Array.isArray(info?.inputs) ? info.inputs : [];
        const names = inputs.map((input) => input?.name);
        const values = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
        if (values && names.includes("control_type") && !names.includes("control_mode")) {
          const oldType = String(values[names.indexOf("control_type")] || "");
          const mode = oldType.startsWith("姿态") ? "姿态" : oldType.startsWith("深度") ? "深度" : "关闭";
          info = { ...info, widgets_values: [mode, values[names.indexOf("resolution")] ?? 768, "GPU", values[names.indexOf("enabled")] ?? true, true, false, 1.0, 1.0] };
        } else if (values && names.includes("control_mode") && !names.includes("preprocess_device")) {
          // The unified node gained the device widget after resolution. Add a
          // GPU default by widget order so old serialized graphs keep every
          // following value aligned, including optional input sockets.
          const widgetNames = inputs.filter((input) => input?.widget).map((input) => input.name);
          const resolutionWidgetIndex = widgetNames.indexOf("resolution");
          const nextValues = [...values];
          nextValues.splice(resolutionWidgetIndex >= 0 ? resolutionWidgetIndex + 1 : 2, 0, "GPU");
          const nextInputs = [...inputs];
          const resolutionInputIndex = nextInputs.findIndex((input) => input?.name === "resolution");
          nextInputs.splice(resolutionInputIndex >= 0 ? resolutionInputIndex + 1 : 0, 0,
            { label: "预处理设备", localized_name: "预处理设备", name: "preprocess_device", type: "COMBO", widget: { name: "preprocess_device" } });
          info = { ...info, inputs: nextInputs, widgets_values: nextValues };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === TRANSFER_LOADER_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        // The hybrid controls were inserted after Ref2VA. Convert the former
        // six-widget transfer loader schema without shifting CLIP or either VAE.
        const names = Array.isArray(info?.inputs) ? info.inputs.map((input) => input?.name) : [];
        if (!names.includes("base_model") && Array.isArray(info?.widgets_values)) {
          const old = info.widgets_values;
          info = {
            ...info,
            widgets_values: [old[0], "MiniMax-H3\\minimax_h3_fl2va_pruned_int8_convrot.safetensors", false,
              old[1], old[2], old[3], old[4] ?? "default", old[5] ?? "minimax"],
          };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === DUAL_STAGE_LOADER_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        // The loader only owns model selection. Remove the retired sampling
        // and Sigma sockets from old serialized workflows before LiteGraph
        // restores them, including stale outputs that were never connected.
        const inputs = Array.isArray(info?.inputs) ? info.inputs : [];
        const names = inputs.map((input) => input?.name);
        const retired = new Set(["sampling_mode", "shift_video", "shift_audio", "stage1_sigmas", "stage2_sigmas"]);
        const oldValues = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
        const hasRetired = names.some((name) => retired.has(name))
          || (Array.isArray(info?.outputs) && info.outputs.some((output) => retired.has(output?.name)));
        if (hasRetired) {
          const values = Object.fromEntries(names.map((name, index) => [name, oldValues?.[index]]));
          const keep = ["stage1_model", "stage1_base_model", "stage1_enable_hybrid",
            "stage2_model", "stage2_base_model", "stage2_enable_hybrid", "weight_dtype"];
          info = {
            ...info,
            inputs: inputs.filter((input) => !retired.has(input?.name)),
            outputs: Array.isArray(info.outputs)
              ? info.outputs.filter((output) => !retired.has(output?.name))
              : info.outputs,
            widgets_values: keep.map((name) => values[name]),
          };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === SAMPLING_SWITCH_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        // The current switch keeps the optional MODEL input for its original
        // SIGMAS branch, but owns neither scheduler controls nor stage steps.
        const names = Array.isArray(info?.inputs) ? info.inputs.map((input) => input?.name) : [];
        const values = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
        const retired = new Set(["scheduler", "steps", "denoise"]);
        const hasLegacyPorts = names.some((name) => retired.has(name));
        if (values && hasLegacyPorts) {
          info = {
            ...info,
            inputs: (info.inputs || []).filter((input) => !retired.has(input?.name)),
            outputs: [
              { name: "采样调度信息", label: "采样调度信息", type: "H3_AUDIO_SAMPLING" },
              { name: "SIGMAS", label: "SIGMAS", type: "SIGMAS" },
            ],
            // MODEL is an input socket, not a widget. Serialized values have
            // always been [sampling_mode, shift_video, shift_audio].
            widgets_values: [values[0] ?? "ComfyUI v0.31.0版本方法", values[1] ?? 12, values[2] ?? 3],
          };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === DUAL_SAMPLING_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        // The old dual-sampling schema did not include the two stage-control
        // booleans.  Its trailing values were therefore read as
        // [stage2_steps, stage2_denoise, upscale_mode, seed], which maps an
        // old seed into stage2_denoise after the controls were added.
        const names = Array.isArray(info?.inputs) ? info.inputs.map((input) => input?.name) : [];
        const values = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
        if (names.includes("enable_stage2") && values?.length === 11
          && Number(values[7]) > 1 && typeof values[8] === "string") {
          info = {
            ...info,
            widgets_values: [
              values[0], values[1], values[2], values[3],
              true, false, values[4], values[5], values[6], values[7], values[8], values[9], values[10],
            ],
          };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === H3_RESOLUTION_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        // Convert the first dual-resolution schema (preset + two integer
        // fields) to the current two-switch and single "width,height" input.
        const names = Array.isArray(info?.inputs) ? info.inputs.map((input) => input?.name) : [];
        if (names.includes("custom_ratio_width") && Array.isArray(info.widgets_values)) {
          const values = Object.fromEntries(names.map((name, index) => [name, info.widgets_values[index]]));
          const legacyPreset = String(values.aspect_preset || "16:9").split(" ")[0];
          const custom = `${values.custom_ratio_width ?? 16},${values.custom_ratio_height ?? 9}`;
          const usesCustom = legacyPreset === "自定义";
          info = { ...info, widgets_values: [!usesCustom, usesCustom, usesCustom ? "16:9" : legacyPreset, custom,
            values.stage1_megapixels ?? 0.4, values.stage2_megapixels ?? 0.98, values.multiple ?? 32] };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === CACHED_REFERENCE_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        if (info) {
          const w = info.properties?.h3_node_size?.[0] || info.size?.[0] || 0;
          const h = info.properties?.h3_node_size?.[1] || info.size?.[1] || 0;
          info.size = [Math.max(Number(w) || 0, 350), Math.max(Number(h) || 0, 310)];
        }
        const res = originalConfigure?.call(this, info);
        if (Array.isArray(this.size)) {
          this.size[0] = Math.max(this.size[0], 350);
          this.size[1] = Math.max(this.size[1], 310);
        }
        return res;
      };
      const originalCompute = nodeType.prototype.computeSize;
      nodeType.prototype.computeSize = function (...args) {
        const sz = originalCompute ? originalCompute.apply(this, args) : [350, 310];
        const savedW = this.properties?.h3_node_size?.[0] || 0;
        const savedH = this.properties?.h3_node_size?.[1] || 0;
        return [Math.max(sz[0] || 0, savedW, 350), Math.max(sz[1] || 0, savedH, 310)];
      };
    }
    if (H3_NODE_CLASSES.has(nodeData.name)) {
      const original = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function (...args) {
        const result = original?.apply(this, args);
        const refreshLabels = () => { decorateNode(this); applyChineseLabels(this); };
        requestAnimationFrame(refreshLabels);
        setTimeout(refreshLabels, 80);
        setTimeout(refreshLabels, 300);
        return result;
      };
    }
    if (nodeData.name !== "MiniMaxH3ReferenceToVideo") return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = original?.apply(this, args);
      requestAnimationFrame(() => requestAnimationFrame(() => removeEmptyReferenceSocket(this)));
      return result;
    };
  },
  nodeCreated(node) {
    decorateNode(node);
    requestAnimationFrame(() => applyChineseLabels(node));
  },
  loadedGraphNode(node) {
    decorateNode(node);
    requestAnimationFrame(() => { decorateNode(node); applyChineseLabels(node); });
    setTimeout(() => applyChineseLabels(node), 120);
  },
});
