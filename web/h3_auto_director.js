import { app } from "../../scripts/app.js";

const NODE = "H3AutoDirectorPlan";
const TRANSFER_NODE = "H3AutoDirectorVideoTransferPlan";
const TRANSFER_LOADER_NODE = "H3AutoDirectorTransferModelLoader";
const HYBRID_LOADER_NODE = "H3AutoDirectorHybridModelLoader";
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
const H3_NODE_CLASSES = new Set([NODE, TRANSFER_NODE, TRANSFER_LOADER_NODE, HYBRID_LOADER_NODE, TTS_NODE, SEGMENT_NODE, REFERENCE_NODE, CACHED_REFERENCE_NODE, DUAL_SAMPLING_NODE, AV_DECODE_NODE, CONTEXT_NODE, RESUME_NODE, MOTION_CONTEXT_NODE, MOTION_TRIM_NODE, MOTION_SAVE_LATENT_NODE, MOTION_LOAD_LATENT_NODE, RESOLUTION_NODE, H3_RESOLUTION_NODE, SAVE_NODE, CONTROLLER_NODE, SAMPLING_SWITCH_NODE]);
const MAX_REFS = { image: 9, video: 3, audio: 3 };
const MAX_TOTAL_REFS = 12;
const DIR_KEY = "h3-auto-director-picker-dirs";
const PICKER_OPTIONS_KEY = "h3-auto-director-picker-options";
const UPLOAD_DIRS = { image: "h3_refs/images", video: "h3_refs/videos", audio: "h3_refs/audio" };
const VIDEO_AUDIO_PROBES = new WeakSet();

function widget(node, name) {
  return (node.widgets || []).find((item) => item.name === name);
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
  const normalized = {
    prompt: String(seg.prompt || ""),
    duration: Number(seg.duration) || 5,
    audio_filename: String(seg.audio_filename || ""),
    audio_restart: !!seg.audio_restart,
    continue_audio: seg.continue_audio !== false,
    continue_video: seg.continue_video !== false,
    references: Array.isArray(seg.references) ? seg.references.map((ref) => {
      if (typeof ref === "string") return { type: "image", name: ref, path: ref };
      const normalized = { ...ref };
      // Image references condition the whole generated segment. They do not
      // have an independent duration; remove legacy values on load/save.
      if (normalized.type === "image") delete normalized.duration;
      else normalized.duration = Number(ref?.duration) > 0 ? Number(ref.duration) : 1;
      return normalized;
    }) : [],
  };
  if (Object.prototype.hasOwnProperty.call(seg, "use_previous_video_reference")) {
    normalized.use_previous_video_reference = !!seg.use_previous_video_reference;
  }
  return normalized;
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
  node.setDirtyCanvas(true, true);
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
  return result.has_audio === true;
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
    ref.has_audio = await probeVideoAudio(path).catch(() => false);
    ref.video_audio_enabled = true;
  }
  return ref;
}

function countRefs(segment, type) {
  return (segment.references || []).filter((ref) => ref.type === type).length;
}

function totalRefs(segment) {
  return (segment.references || []).length;
}

function videoAudioRefs(segment) {
  return (segment.references || []).filter((ref) => ref.type === "video" && ref.has_audio === true && ref.video_audio_enabled !== false);
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

function applyChineseLabels(node) {
  const nodeClass = node.comfyClass || node.type;
  if (!H3_NODE_CLASSES.has(nodeClass)) return;
  const labels = {
    project: "项目计划", project_id: "总文件夹名称", segments_json: "片段配置", duration: "默认片段时长",
    global_reference_set: "统一参考集", auto_run: "自动连续生成", continuation_mode: "接续模式",
    cache_prompt_embeddings: "一次性缓存提示词向量", global_assets_json: "统一参考素材",
    segment_index: nodeClass === SEGMENT_NODE || nodeClass === CONTEXT_NODE || nodeClass === RESUME_NODE ? "上下文片段序号" : "片段序号",
    context_length: "上下文长度", prompt: "提示词", references_json: "参考素材 JSON",
    clip: "文本编码器", vae: "视频 VAE", audio_vae: "音频 VAE", width: "宽度", height: "高度", length: "帧数",
    ref_image_size: "参考图片尺寸", enable_resume: "启用断点续接", latent_path: "缓存潜变量路径", video_path: "缓存视频路径",
    conditioning: "条件", latent: "潜变量", context_frames: "上下文画面", context_latent: "上下文潜变量",
    use_video_context: "使用视频上下文", use_audio_context: "使用音频上下文", use_video_latent: "使用视频潜空间",
    fps: "帧率", images: "视频画面",
    audio: "音频", saved_video: nodeClass === "H3AutoDirectorTTSController" ? "已保存音频" : "已保存视频", segment_node_id: "片段节点 ID", trim_frames: "裁剪帧数", match_tail: "匹配音频尾部",
    clip_index: "片段序号", latent_path: "潜变量路径",
    aspect_ratio: "宽高比", megapixels: "目标像素数（MP）", multiple: "尺寸倍数",
    use_preset_ratio: "使用预设比例", use_custom_ratio: "使用自定义比例", aspect_preset: "宽高比预设", custom_ratio: "自定义比例（宽,高）",
    stage1_megapixels: "第一阶段像素数（MP）", stage2_megapixels: "第二阶段像素数（MP）",
    resolution_preview: "当前输出分辨率",
    output_root: nodeClass === SAVE_NODE ? "输出文件名（中间片段，留空使用 H3）" : nodeClass === CONTROLLER_NODE ? "输出文件名（最终视频，留空使用 H3）" : nodeClass === "H3AutoDirectorTTSController" ? "最终长 WAV 文件名（留空使用 H3）" : "项目文件夹名称（保存于 output/h3_project 下）",
    video_format: "视频格式", video_codec: "编码格式", encoder_device: "编码设备", quality: "编码质量", color_correction: "上下文色彩校正",
    scene_cut_protection: "场景切换保护", scene_cut_threshold: "场景切换阈值",
    correction_strength: "校色强度", residual_strength: "残余漂移强度",
    cleanup_after_final: "最终完成后清理显存", sampling_mode: "音频采样切换",
    stage1_steps: "第一阶段步数", stage1_denoise: "第一阶段降噪", enable_stage2: "启用第二阶段采样", stage2_use_context: "二采使用上下文接续", stage2_steps: "第二阶段步数", stage2_denoise: "第二阶段降噪",
    upscale_mode: "视频放大方式", target_width: "第二阶段宽度", target_height: "第二阶段高度", upscale_model: "普通放大模型", seed: "双采样种子",
    shift_video: "视频调度偏移", shift_audio: "音频调度偏移",
    unet_name: nodeClass === HYBRID_LOADER_NODE || nodeClass === TRANSFER_LOADER_NODE ? "多模态参考模型（Ref2VA）" : "扩散模型",
    base_model: "画面基础模型（FL2VA）", enable_hybrid: "启用 H3 混合模型", weight_dtype: "权重数据类型",
    reference_video_json: "参考动作视频", reference_assets_json: "附加参考素材",
    segment_seconds: "每段秒数", pass_reference_video_audio: "传递参考视频音频",
    enable_audio_continuation: "开启音频上下文接续", audio_restart_segments: "重新生成音频片段",
    previous_video_reference_segments: "使用上段视频参考片段", skip_h3_audio_decode: "仅不解码 H3 音频（仍联合采样）",
    final_audio_source: "最终视频音频来源", edit_transfer: "编辑动作迁移计划",
    concat_final_audio: "拼接最终长音频", edit_tts: "编辑 TTS 片段",
  };
  const apply = (item) => {
    const label = labels[item?.name];
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
  if (nodeClass === H3_RESOLUTION_NODE) decorateH3Resolution(node);
  const labels = {
    project_id: "总文件夹名称",
    segments_json: "片段配置",
    duration: "默认片段时长",
    global_reference_set: "统一参考集",
    auto_run: "自动连续生成",
    continuation_mode: "接续模式",
    cache_prompt_embeddings: "一次性缓存提示词向量",
    global_assets_json: "统一参考素材",
    output_root: nodeClass === SAVE_NODE ? "输出文件名（中间片段，留空使用 H3）" : nodeClass === CONTROLLER_NODE ? "输出文件名（最终视频，留空使用 H3）" : "项目文件夹名称（保存于 output/h3_project 下）",
    video_format: "视频格式",
    video_codec: "编码格式",
    encoder_device: "编码设备",
    quality: "编码质量",
    color_correction: "上下文色彩校正",
    use_video_latent: "使用视频潜空间",
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
  if (nodeClass === SAVE_NODE) {
    const output = widget(node, "output_root");
    // Migrate the old directory default without touching the workflow file.
    if (output && String(output.value || "").trim() === "h3_projects") { output.value = ""; output.callback?.(output.value); }
  }
  normalizeSaveFps(node);
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
  const alignment = Math.max(1, Math.trunc(Number(multiple) || 32));
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
  const set = (name, value) => { const item = widget(node, name); if (item) { item.value = value; item.callback?.(value); } };
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
  const panel = document.createElement("div"); panel.style.cssText = "width:min(1120px,95vw);height:min(860px,92vh);max-height:92vh;box-sizing:border-box;resize:both;overflow:hidden;background:#20252b;color:#eee;border:1px solid #59636e;border-radius:8px;padding:18px;display:flex;flex-direction:column";
  const title = document.createElement("h2"); title.textContent = "H3 自动导演｜TTS 片段计划"; title.style.margin = "0 0 12px"; panel.appendChild(title);
  const notice = document.createElement("div"); notice.style.cssText = "color:#aeb7c1;font-size:12px;min-height:22px"; panel.appendChild(notice);
  const settings = document.createElement("div"); settings.style.cssText = "display:flex;flex-wrap:wrap;gap:12px;padding:10px;background:#15191d;border:1px solid #424b55;border-radius:6px;margin-bottom:12px"; panel.appendChild(settings);
  const checkbox = (label, name, fallback) => { const wrap = document.createElement("label"); wrap.style.cssText = "display:flex;align-items:center;gap:6px"; const input = document.createElement("input"); input.type = "checkbox"; input.checked = !!get(name, fallback); wrap.append(input, label); settings.appendChild(wrap); return input; };
  const unified = checkbox("统一参考集（所有片段使用第 1 段素材）", "global_reference_set", false);
  const concat = checkbox("拼接最终长音频", "concat_final_audio", true);
  const continuation = checkbox("开启音频上下文接续", "enable_audio_continuation", true);
  const cache = checkbox("一次性缓存提示词向量", "cache_prompt_embeddings", true);
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
      const controls = document.createElement("div"); controls.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:7px";
      const duration = document.createElement("input"); duration.type = "number"; duration.min = "4"; duration.max = "15"; duration.step = "0.1"; duration.value = Number(seg.duration) || 5; duration.style.width = "80px"; duration.oninput = () => { seg.duration = Number(duration.value) || 5; };
      const filename = document.createElement("input"); filename.type = "text"; filename.value = String(seg.audio_filename || ""); filename.placeholder = `H3_${String(index + 1).padStart(5, "0")}.wav`; filename.style.cssText = "width:250px;background:#20252b;color:#eee;border:1px solid #59636e;padding:5px"; filename.oninput = () => { seg.audio_filename = filename.value; };
      controls.append("秒数", duration, "音频文件名", filename); if (segments.length > 1) controls.appendChild(makeButton("删除片段", () => { segments.splice(index, 1); render(); })); box.appendChild(controls);
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
          const row = document.createElement("div"); row.style.cssText = "display:flex;align-items:center;gap:8px;padding:6px;background:#15191d;border:1px solid #424b55;border-radius:4px";
          row.append(refLabel(ref, type, refs)); if (type === "video" && ref.has_audio === true) { const toggle = document.createElement("label"); toggle.style.cssText = "margin-left:auto;display:flex;align-items:center;gap:4px;font-size:12px"; const input = document.createElement("input"); input.type = "checkbox"; input.checked = ref.video_audio_enabled !== false; input.disabled = !editable; input.onchange = () => { ref.video_audio_enabled = input.checked; renderRefs(); }; toggle.append(input, "传递音频"); row.appendChild(toggle); }
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
  const actions = document.createElement("div"); actions.style.cssText = "flex:0 0 auto;display:flex;justify-content:flex-end;gap:8px;margin:14px -18px -18px;padding:12px 18px;background:rgba(32,37,43,.98);border-top:1px solid #59636e";
  actions.append(makeButton("+ 添加片段", () => { segments.push({ prompt: "", duration: 5, audio_filename: "", audio_restart: false, continue_audio: continuation.checked, references: [], _media_references_open: false, _audio_references_open: true }); render(); }), makeButton("将第 1 段参考素材应用到全部", () => { const refs = (segments[0].references || []).map((ref) => ({ ...ref })); segments.forEach((seg) => { seg.references = refs.map((ref) => ({ ...ref })); }); unified.checked = true; render(); }), makeButton("取消", () => shade.remove()), makeButton("保存", () => { const names = new Set(); for (const seg of segments) { if (Number(seg.duration) < 4 || Number(seg.duration) > 15) { notice.textContent = "每段时长必须在 4 到 15 秒之间。"; return; } const name = String(seg.audio_filename || "").trim(); if (name && !name.toLowerCase().endsWith(".wav")) { notice.textContent = "音频文件名必须使用 .wav 扩展名。"; return; } if (name && names.has(name.toLowerCase())) { notice.textContent = `音频文件名重复：${name}`; return; } if (name) names.add(name.toLowerCase()); } if (unified.checked) { const refs = (segments[0].references || []).map((ref) => ({ ...ref })); segments.forEach((seg) => { seg.references = refs.map((ref) => ({ ...ref })); }); } const savedSegments = segments.map((seg) => { const copy = { ...seg }; delete copy._media_references_open; delete copy._audio_references_open; return copy; }); set("segments_json", JSON.stringify(savedSegments, null, 2)); set("global_reference_set", unified.checked); set("cache_prompt_embeddings", cache.checked); set("enable_audio_continuation", continuation.checked); set("concat_final_audio", concat.checked); node.setDirtyCanvas(true, true); shade.remove(); }));
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
  const set = (name, value) => { const item = widget(node, name); if (item) { item.value = value; item.callback?.(value); } };
  const shade = document.createElement("div");
  shade.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(920px,94vw);height:min(760px,90vh);resize:both;overflow:auto;background:#20252b;color:#eee;border:1px solid #59636e;border-radius:8px;padding:18px;box-shadow:0 16px 60px #000";
  const title = document.createElement("h2"); title.textContent = "H3 自动导演｜动作迁移项目计划"; title.style.margin = "0 0 12px"; panel.appendChild(title);
  const notice = document.createElement("div"); notice.style.cssText = "color:#aeb7c1;font-size:12px;min-height:22px;margin:6px 0"; panel.appendChild(notice);
  const summary = document.createElement("div"); summary.style.cssText = "padding:10px;background:#15191d;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;line-height:1.6"; panel.appendChild(summary);
  const refreshSummary = () => {
    const seconds = Number(get("segment_seconds", 5)) || 5;
    const duration = Number(video.duration || 0);
    const count = duration > 0 ? Math.ceil(duration / seconds) : 0;
    summary.textContent = video.path ? `参考视频：${video.originalName || video.name || video.path}｜时长：${formatDuration(duration)}｜按 ${seconds.toFixed(2)} 秒/段：${count || "等待读取"} 段` : "尚未上传参考视频";
  };
  const row = (label, control) => { const wrap = document.createElement("label"); wrap.style.cssText = "display:flex;align-items:center;gap:8px;margin:8px 0"; wrap.append(label, control); panel.appendChild(wrap); return control; };
  const prompt = document.createElement("textarea"); prompt.value = get("prompt", ""); prompt.style.cssText = "width:100%;min-height:100px;box-sizing:border-box;background:#15191d;color:#eee;border:1px solid #59636e;padding:8px"; prompt.placeholder = "所有片段复用的 H3 完整提示词"; panel.appendChild(prompt);
  const seconds = document.createElement("input"); seconds.type = "number"; seconds.min = "4"; seconds.max = "15"; seconds.step = "0.1"; seconds.value = get("segment_seconds", 5); seconds.style.width = "110px"; seconds.oninput = refreshSummary; row("每段秒数", seconds);
  const checkbox = (label, name, checked) => { const input = document.createElement("input"); input.type = "checkbox"; input.checked = !!get(name, checked); row(label, input); return input; };
  const passAudio = checkbox("传递参考视频音频", "pass_reference_video_audio", false);
  const audioCont = checkbox("开启音频上下文接续", "enable_audio_continuation", true);
  const cachePrompts = checkbox("一次性缓存全部片段的提示词向量", "cache_prompt_embeddings", true);
  const autoRun = checkbox("自动连续生成并在最后拼接", "auto_run", true);
  const skipDecode = checkbox("仅不解码 H3 音频（仍联合采样）", "skip_h3_audio_decode", false);
  const audioMode = document.createElement("select"); audioMode.innerHTML = "<option>H3 生成音频</option><option>参考视频音频</option>"; audioMode.value = get("final_audio_source", "H3 生成音频"); row("最终视频音频来源", audioMode);
  const restart = document.createElement("input"); restart.type = "text"; restart.value = get("audio_restart_segments", ""); restart.placeholder = "例如 3，6,9"; restart.style.width = "220px"; row("重新生成音频片段", restart);
  const previous = document.createElement("input"); previous.type = "text"; previous.value = get("previous_video_reference_segments", ""); previous.placeholder = "例如 2,5"; previous.style.width = "220px"; row("使用上段视频参考片段", previous);
  const assetList = document.createElement("div"); assetList.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;margin:12px 0"; panel.appendChild(assetList);
  const renderAssets = () => { assetList.replaceChildren(); assets.forEach((asset, index) => { const item = document.createElement("div"); item.style.cssText = "padding:8px;background:#15191d;border:1px solid #424b55;border-radius:5px;display:flex;justify-content:space-between;gap:8px"; item.append(`${asset.type === "image" ? "图片" : "音频"}${index + 1}：${asset.originalName || asset.name}`); const remove = makeButton("删除", () => { assets.splice(index, 1); renderAssets(); }); item.appendChild(remove); assetList.appendChild(item); }); };
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
      refreshSummary(); notice.textContent = "参考视频已上传；分段数量会按视频时长和每段秒数自动计算。";
    } catch (error) { notice.textContent = error.message || String(error); }
  };
  panel.appendChild(makeButton(video.path ? "重新上传参考视频" : "+ 上传参考视频", uploadVideo));
  const actions = document.createElement("div"); actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;margin-top:16px";
  actions.append(makeButton("取消", () => shade.remove()), makeButton("保存", () => {
    if (!video.path) { notice.textContent = "请先上传参考视频。"; return; }
    set("prompt", prompt.value); set("segment_seconds", Number(seconds.value) || 5); set("pass_reference_video_audio", passAudio.checked); set("enable_audio_continuation", audioCont.checked); set("cache_prompt_embeddings", cachePrompts.checked); set("auto_run", autoRun.checked); set("skip_h3_audio_decode", skipDecode.checked); set("final_audio_source", audioMode.value); set("audio_restart_segments", restart.value); set("previous_video_reference_segments", previous.value); set("reference_video_json", JSON.stringify(video, null, 2)); set("reference_assets_json", JSON.stringify(assets, null, 2)); node.setDirtyCanvas(true, true); shade.remove();
  }));
  panel.appendChild(actions); shade.appendChild(panel); document.body.appendChild(shade); renderAssets(); refreshSummary();
}

function openEditor(node) {
  let segments = readSegments(node);
  if (!segments.length) segments = [{ prompt: "", duration: 5, audio_restart: false, references: [] }];
  const dirs = readDirectories();
  const pickerOptions = readPickerOptions();
  const pickerHandles = {};

  const shade = document.createElement("div");
  shade.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(1240px,96vw);height:min(88vh,900px);min-width:760px;min-height:520px;resize:both;overflow:auto;background:#20252b;color:#eee;border:1px solid #59636e;border-radius:8px;padding:20px;box-shadow:0 16px 60px #000;display:flex;flex-direction:column";
  const title = document.createElement("h2");
  title.textContent = "H3 自动导演｜片段列表";
  title.style.cssText = "margin:0 0 12px;font-size:22px";
  panel.appendChild(title);

  const dirPanel = document.createElement("div");
  dirPanel.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px";
  const dirTitle = document.createElement("div");
  dirTitle.textContent = "默认打开路径只影响文件选择器，导入后的素材固定保存到 h3_refs/*。Python 模式在 ComfyUI 所在机器打开系统对话框。";
  dirTitle.style.cssText = "grid-column:1/-1;font-size:12px;color:#aeb7c1";
  dirPanel.appendChild(dirTitle);
  const pickerControls = document.createElement("div");
  pickerControls.style.cssText = "grid-column:1/-1;display:flex;align-items:center;gap:14px;font-size:12px";
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
  const directActions = document.createElement("div"); directActions.style.cssText = "display:flex;gap:6px;margin-top:7px";
  directActions.appendChild(makeButton("应用 JSON", () => {
    try {
      const parsed = JSON.parse(directInput.value);
      const next = normalizeSegments(parsed);
      if (!next.length) throw new Error("片段配置必须是非空数组");
      if (!confirmSegmentReduction(segments, next.length)) { notice.textContent = "已取消减少片段。"; return; }
      if (next.some((seg) => seg.duration < 4 || seg.duration > 15)) throw new Error("片段时长必须在 4～15 秒之间");
      if (next.some((seg) => totalRefs(seg) > MAX_TOTAL_REFS || Object.entries(MAX_REFS).some(([type, max]) => countRefs(seg, type) > max))) throw new Error(`每段参考素材最多 ${MAX_TOTAL_REFS} 个`);
      segments = next; directInput.value = JSON.stringify(segments, null, 2); segmentCountInput.value = segments.length; render(); notice.textContent = "已应用直接输入的片段配置。";
    } catch (error) { notice.textContent = `JSON 无效：${error.message || error}`; }
  }));
  directActions.appendChild(makeButton("从列表更新 JSON", () => { directInput.value = JSON.stringify(segments, null, 2); }));
  directPanel.appendChild(directActions); panel.appendChild(directPanel);

  const segmentDurationPanel = document.createElement("div"); segmentDurationPanel.style.cssText = "display:flex;align-items:center;gap:8px;padding:8px 10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;font-size:12px";
  segmentDurationPanel.append("统一片段秒数");
  const segmentSeconds = document.createElement("input"); segmentSeconds.type = "number"; segmentSeconds.min = "4"; segmentSeconds.max = "15"; segmentSeconds.step = "0.1"; segmentSeconds.value = "5"; segmentSeconds.style.cssText = "width:76px;background:#15191d;color:#eee;border:1px solid #59636e;padding:5px"; segmentDurationPanel.appendChild(segmentSeconds);
  segmentDurationPanel.appendChild(makeButton("应用到全部片段", () => {
    const value = Math.max(4, Math.min(15, Number(segmentSeconds.value) || 5));
    segmentSeconds.value = String(value);
    segments.forEach((seg) => { seg.duration = value; });
    directInput.value = JSON.stringify(segments, null, 2);
    render();
    notice.textContent = `已将全部片段的秒数设置为 ${value}。`;
  }));
  panel.appendChild(segmentDurationPanel);

  const segmentCountPanel = document.createElement("div"); segmentCountPanel.style.cssText = "display:flex;align-items:center;gap:8px;padding:8px 10px;background:#171b20;border:1px solid #424b55;border-radius:6px;margin-bottom:12px;font-size:12px";
  segmentCountPanel.append("片段数量");
  const segmentCountInput = document.createElement("input"); segmentCountInput.type = "number"; segmentCountInput.min = "1"; segmentCountInput.max = "999"; segmentCountInput.step = "1"; segmentCountInput.value = String(segments.length); segmentCountInput.style.cssText = "width:76px;background:#15191d;color:#eee;border:1px solid #59636e;padding:5px"; segmentCountPanel.appendChild(segmentCountInput);
  segmentCountPanel.appendChild(makeButton("应用片段数量", () => {
    const target = Math.max(1, Math.min(999, Math.floor(Number(segmentCountInput.value) || segments.length)));
    if (!confirmSegmentReduction(segments, target)) { segmentCountInput.value = String(segments.length); notice.textContent = "已取消减少片段。"; return; }
    while (segments.length < target) segments.push({ prompt: "", duration: 5, audio_restart: false, continue_audio: true, continue_video: true, references: [], _references_open: false });
    if (segments.length > target) segments.length = target;
    segmentCountInput.value = String(target); directInput.value = JSON.stringify(segments, null, 2); render(); notice.textContent = `片段数量已设置为 ${target}。`;
  }));
  panel.appendChild(segmentCountPanel);

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
    return videos.slice(0, videos.indexOf(ref) + 1).filter((item) => item.has_audio === true && item.video_audio_enabled !== false).length;
  };

  const createRefCard = (seg, ref, type, renderRefs) => {
    const promptNumber = referencePromptNumber(seg, ref, type);
    const promptTag = `<${type === "image" ? "Picture" : type === "video" ? "Video" : "Audio"} ${promptNumber}>`;
    const mediaDetails = () => `提示词标签：${promptTag} | 时长：${formatDuration(ref.duration)}`;
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
          }).catch(() => { ref.has_audio = false; renderRefs(); });
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
    title.textContent = `视频音频 ${number}`;
    title.style.cssText = "font-weight:700;color:#f0f3f6";
    info.appendChild(title);
    const meta = document.createElement("div");
    meta.textContent = `提示词标签：<Audio ${number}> | 来源：${videoRef.originalName || videoRef.name || "视频参考"}`;
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
      row.style.cssText = "margin:8px 0;padding:10px;background:#2b3138;border:1px solid #424b55;border-radius:6px";
      const head = document.createElement("div"); head.style.cssText = "display:grid;grid-template-columns:36px minmax(220px,1fr) 92px 140px 210px 230px 32px;gap:8px;align-items:start";
      const label = document.createElement("strong"); label.textContent = String(index + 1); label.style.paddingTop = "7px"; head.appendChild(label);
      const prompt = document.createElement("textarea"); prompt.rows = 3; prompt.placeholder = "输入本片段提示词，使用 <Picture 1>/<Video 1>/<Audio 1> 标签"; prompt.value = seg.prompt; prompt.style.cssText = "width:100%;resize:vertical;background:#15191d;color:#eee;border:1px solid #59636e;padding:6px"; prompt.oninput = () => { seg.prompt = prompt.value; }; head.appendChild(prompt);
      const duration = document.createElement("input"); duration.type = "number"; duration.min = "4"; duration.max = "15"; duration.step = "0.1"; duration.value = seg.duration; duration.title = "片段时长（秒）"; duration.style.cssText = "width:84px;background:#15191d;color:#eee;border:1px solid #59636e;padding:6px"; duration.oninput = () => { seg.duration = Number(duration.value) || 5; }; head.appendChild(duration);
      const reset = document.createElement("label"); reset.style.cssText = "font-size:12px;display:flex;gap:5px;align-items:center;padding-top:7px"; const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.checked = seg.audio_restart; checkbox.onchange = () => { seg.audio_restart = checkbox.checked; }; reset.append(checkbox, "重新生成音频"); head.appendChild(reset);
      const videoReset = document.createElement("label"); videoReset.style.cssText = "font-size:12px;display:flex;gap:5px;align-items:center;padding-top:7px";
      const videoResetCheckbox = document.createElement("input"); videoResetCheckbox.type = "checkbox"; videoResetCheckbox.checked = index > 0 && seg.continue_video === false; videoResetCheckbox.disabled = index === 0;
      videoResetCheckbox.onchange = () => { seg.continue_video = !videoResetCheckbox.checked; if (videoResetCheckbox.checked) { previousCheckbox.checked = false; seg.use_previous_video_reference = false; } };
      videoReset.append(videoResetCheckbox, "重新生成视频（关闭视频上下文）"); head.appendChild(videoReset);
      const previous = document.createElement("label"); previous.style.cssText = "font-size:12px;display:flex;gap:5px;align-items:center;padding-top:7px";
      const previousCheckbox = document.createElement("input"); previousCheckbox.type = "checkbox"; previousCheckbox.checked = index > 0 && !!seg.use_previous_video_reference; previousCheckbox.disabled = index === 0;
      previousCheckbox.onchange = () => { seg.use_previous_video_reference = previousCheckbox.checked; if (previousCheckbox.checked) { seg.continue_video = false; videoResetCheckbox.checked = true; } };
      previous.append(previousCheckbox, "使用上片段视频作为参考素材接续"); head.appendChild(previous);
      head.appendChild(makeButton("×", () => {
        if (segments.length <= 1) return;
        if (!confirmSegmentReduction(segments, segments.length - 1, [seg])) { notice.textContent = "已取消删除片段。"; return; }
        segments.splice(index, 1); render();
      }, "删除片段")); row.appendChild(head);

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
            if (type === "video" && ref.has_audio === true) group.appendChild(createVideoAudioCard(seg, ref, renderRefs));
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
  };
  render();

  const actions = document.createElement("div"); actions.style.cssText = "display:flex;gap:8px;justify-content:flex-end;margin-top:12px";
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
    writeSegments(node, segments); shade.remove();
  }));
  panel.appendChild(actions); shade.appendChild(panel); document.body.appendChild(shade);
}

app.registerExtension({
  name: "H3AutoDirector.Editor",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        // Older workflows serialized the removed global previous-video widget
        // between continuation_mode and cache_prompt_embeddings. Drop only
        // that value so all remaining widgets keep their original meaning.
        const legacy = Array.isArray(info?.inputs)
          && info.inputs.some((input) => input?.name === "use_previous_video_reference");
        if (legacy && Array.isArray(info.widgets_values) && info.widgets_values.length >= 10) {
          info = { ...info, widgets_values: info.widgets_values.slice(0, 6).concat(info.widgets_values.slice(7)) };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === TTS_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        // Older TTS workflows serialized global reference/audio widgets. The
        // current node stores those values per segment, so drop the obsolete
        // widget slots while preserving the remaining values by name.
        const inputs = Array.isArray(info?.inputs) ? info.inputs : [];
        const names = inputs.map((input) => input?.name);
        const legacyNames = ["reference_video_json", "reference_assets_json", "pass_reference_video_audio", "audio_restart_segments"];
        if (legacyNames.some((name) => names.includes(name)) && Array.isArray(info.widgets_values)) {
          const keep = ["project_id", "segments_json", "auto_run", "cache_prompt_embeddings", "enable_audio_continuation", "concat_final_audio", "output_root", "global_reference_set"];
          const values = Object.fromEntries(names.map((name, index) => [name, info.widgets_values[index]]));
          info = { ...info, widgets_values: keep.map((name) => values[name]) };
        }
        return originalConfigure?.call(this, info);
      };
    }
    if (nodeData.name === TRANSFER_NODE) {
      const originalConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        // Remove the two retired freeze widgets by name so the values that
        // follow them keep their correct input mapping in old workflows.
        const names = Array.isArray(info?.inputs) ? info.inputs.map((input) => input?.name) : [];
        const retired = ["freeze_video_sampling", "freeze_audio_sampling"];
        if (retired.some((name) => names.includes(name)) && Array.isArray(info.widgets_values)) {
          info = { ...info, widgets_values: info.widgets_values.filter((_, index) => !retired.includes(names[index])) };
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
