import { app } from "../../scripts/app.js";

const NODE = "H3AutoDirectorPlan";
const SEGMENT_NODE = "H3AutoDirectorSegment";
const REFERENCE_NODE = "H3AutoDirectorReferenceResolver";
const CACHED_REFERENCE_NODE = "H3AutoDirectorCachedReferenceToVideo";
const CONTEXT_NODE = "H3AutoDirectorContext";
const RESUME_NODE = "H3AutoDirectorResumeContext";
const MOTION_CONTEXT_NODE = "H3AutoDirectorMotionContext";
const MOTION_TRIM_NODE = "MiniMaxH3MotionContextTrim";
const MOTION_SAVE_LATENT_NODE = "MiniMaxH3MotionContextSaveLatent";
const MOTION_LOAD_LATENT_NODE = "MiniMaxH3MotionContextLoadLatent";
const RESOLUTION_NODE = "ResolutionSelector";
const SAVE_NODE = "H3AutoDirectorSaveSegment";
const CONTROLLER_NODE = "H3AutoDirectorController";
const H3_NODE_CLASSES = new Set([NODE, SEGMENT_NODE, REFERENCE_NODE, CACHED_REFERENCE_NODE, CONTEXT_NODE, RESUME_NODE, MOTION_CONTEXT_NODE, MOTION_TRIM_NODE, MOTION_SAVE_LATENT_NODE, MOTION_LOAD_LATENT_NODE, RESOLUTION_NODE, SAVE_NODE, CONTROLLER_NODE]);
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
  return {
    prompt: String(seg.prompt || ""),
    duration: Number(seg.duration) || 5,
    audio_restart: !!seg.audio_restart,
    continue_video: seg.continue_video !== false,
    references: Array.isArray(seg.references) ? seg.references.map((ref) => {
      if (typeof ref === "string") return { type: "image", name: ref, path: ref, duration: 1 };
      return { ...ref, duration: Number(ref?.duration) > 0 ? Number(ref.duration) : 1 };
    }) : [],
  };
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
  w.value = JSON.stringify(segments, null, 2);
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
    use_video_context: "使用视频上下文", use_audio_context: "使用音频上下文", fps: "帧率", images: "视频画面",
    audio: "音频", saved_video: "已保存视频", segment_node_id: "片段节点 ID", trim_frames: "裁剪帧数", match_tail: "匹配音频尾部",
    clip_index: "片段序号", latent_path: "潜变量路径",
    aspect_ratio: "宽高比", megapixels: "目标像素数（MP）", multiple: "尺寸倍数",
    output_root: nodeClass === SAVE_NODE ? "输出文件名（中间片段，留空使用 H3）" : nodeClass === CONTROLLER_NODE ? "输出文件名（最终视频，留空使用 H3）" : "项目文件夹名称（保存于 output/h3_project 下）",
    video_format: "视频格式", video_codec: "编码格式", encoder_device: "编码设备", quality: "编码质量",
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
  };
  applyChineseLabels(node);
  if (nodeClass === SAVE_NODE) {
    const output = widget(node, "output_root");
    // Migrate the old directory default without touching the workflow file.
    if (output && String(output.value || "").trim() === "h3_projects") { output.value = ""; output.callback?.(output.value); }
  }
  normalizeSaveFps(node);
  if (nodeClass === NODE) node.setSize([900, 360]);
  if (nodeClass === SAVE_NODE || nodeClass === CONTROLLER_NODE) node.setSize([430, 320]);
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
    while (segments.length < target) segments.push({ prompt: "", duration: 5, audio_restart: false, continue_video: true, references: [] });
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
    if (type === "image") {
      const imageDuration = document.createElement("label"); imageDuration.textContent = "图片秒数"; imageDuration.style.cssText = "display:flex;align-items:center;gap:6px;font-size:11px;color:#aeb7c1";
      const seconds = document.createElement("input"); seconds.type = "number"; seconds.min = "0.1"; seconds.max = "60"; seconds.step = "0.1"; seconds.value = Number(ref.duration) || 1; seconds.style.cssText = "width:70px;background:#0d1013;color:#ddd;border:1px solid #424b55;padding:3px";
      seconds.oninput = () => { ref.duration = Math.max(0.1, Number(seconds.value) || 1); };
      imageDuration.appendChild(seconds); info.appendChild(imageDuration);
    }
    card.appendChild(info);
    card.appendChild(makeButton("×", () => { seg.references.splice(seg.references.indexOf(ref), 1); renderRefs(); }, "删除素材"));
    return card;
  };

  const createVideoAudioCard = (seg, videoRef, renderRefs) => {
    const number = videoAudioPromptNumber(seg, videoRef);
    const card = document.createElement("div");
    card.style.cssText = "display:grid;grid-template-columns:1fr 30px;gap:8px;align-items:center;padding:8px;background:#171b20;border:1px solid #424b55;border-radius:5px";
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
    toggle.style.cssText = "display:flex;align-items:center;gap:5px;font-size:11px;color:#d6dde5;white-space:nowrap";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = videoRef.video_audio_enabled !== false;
    checkbox.title = "关闭后只传递视频画面";
    checkbox.onchange = () => { videoRef.video_audio_enabled = checkbox.checked; renderRefs(); };
    toggle.append(checkbox, "传递音频");
    card.append(info, toggle);
    return card;
  };

  const render = () => {
    segmentCountInput.value = String(segments.length);
    list.innerHTML = "";
    segments.forEach((seg, index) => {
      const row = document.createElement("div");
      row.style.cssText = "margin:8px 0;padding:10px;background:#2b3138;border:1px solid #424b55;border-radius:6px";
      const head = document.createElement("div"); head.style.cssText = "display:grid;grid-template-columns:36px minmax(220px,1fr) 92px 140px 32px;gap:8px;align-items:start";
      const label = document.createElement("strong"); label.textContent = String(index + 1); label.style.paddingTop = "7px"; head.appendChild(label);
      const prompt = document.createElement("textarea"); prompt.rows = 3; prompt.placeholder = "输入本片段提示词，使用 <Picture 1>/<Video 1>/<Audio 1> 标签"; prompt.value = seg.prompt; prompt.style.cssText = "width:100%;resize:vertical;background:#15191d;color:#eee;border:1px solid #59636e;padding:6px"; prompt.oninput = () => { seg.prompt = prompt.value; }; head.appendChild(prompt);
      const duration = document.createElement("input"); duration.type = "number"; duration.min = "4"; duration.max = "15"; duration.step = "0.1"; duration.value = seg.duration; duration.title = "片段时长（秒）"; duration.style.cssText = "width:84px;background:#15191d;color:#eee;border:1px solid #59636e;padding:6px"; duration.oninput = () => { seg.duration = Number(duration.value) || 5; }; head.appendChild(duration);
      const reset = document.createElement("label"); reset.style.cssText = "font-size:12px;display:flex;gap:5px;align-items:center;padding-top:7px"; const checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.checked = seg.audio_restart; checkbox.onchange = () => { seg.audio_restart = checkbox.checked; }; reset.append(checkbox, "重新生成音频"); head.appendChild(reset);
      head.appendChild(makeButton("×", () => {
        if (segments.length <= 1) return;
        if (!confirmSegmentReduction(segments, segments.length - 1, [seg])) { notice.textContent = "已取消删除片段。"; return; }
        segments.splice(index, 1); render();
      }, "删除片段")); row.appendChild(head);

      const details = document.createElement("details"); details.open = (seg.references || []).length > 0; details.style.cssText = "margin-top:8px;background:#20252b;border:1px solid #59636e;border-radius:5px;padding:6px 8px";
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
  actions.appendChild(makeButton("+ 添加片段", () => { segments.push({ prompt: "", duration: 5, audio_restart: false, references: [] }); render(); }));
  actions.appendChild(makeButton("将第 1 段参考素材应用到全部", () => { const refs = JSON.parse(JSON.stringify(segments[0].references || [])); segments.forEach((seg) => { seg.references = JSON.parse(JSON.stringify(refs)); }); render(); }));
  actions.appendChild(makeButton("取消", () => shade.remove()));
  actions.appendChild(makeButton("保存", () => {
    const invalid = segments.findIndex((seg) => totalRefs(seg) > MAX_TOTAL_REFS || Object.entries(MAX_REFS).some(([type, max]) => countRefs(seg, type) > max));
    if (invalid >= 0) { notice.textContent = `第 ${invalid + 1} 段参考素材数量超过限制（每段最多 ${MAX_TOTAL_REFS} 个）。`; return; }
    writeSegments(node, segments); shade.remove();
  }));
  panel.appendChild(actions); shade.appendChild(panel); document.body.appendChild(shade);
}

app.registerExtension({
  name: "H3AutoDirector.Editor",
  beforeRegisterNodeDef(nodeType, nodeData) {
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
