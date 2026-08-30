# ComfyUI H3 自动导演

本插件为 MiniMax H3 提供多片段项目计划、参考素材管理、连续生成、AV latent 缓存、上下文接续、双阶段采样和最终拼接节点。

## 当前节点

- `H3 自动导演｜项目计划`：保存片段提示词、时长、参考素材、连续生成和缓存设置。
- `H3 自动导演｜TTS 项目计划`：按片段生成和保存音频；可独立控制音频接续和最终音频拼接。
- `H3 自动导演｜动作迁移项目计划`：按参考动作视频切分片段，视频可选择仅用于动作/时长计算或同时作为多模态参考。
- `H3 自动导演｜片段设置`、`H3 自动导演｜多模态素材解析`、`H3 自动导演｜提示词与素材缓存`：为当前片段准备提示词、参考图像/视频/音频和缓存条件。
- `H3 自动导演｜上下文读取`、`H3 自动导演｜断点续接`、`H3 自动导演｜运动上下文`：读取项目中的视频、音频或 AV latent 上下文。
- `H3 自动导演｜双阶段采样`：可关闭二采；二采模型未连接时复用一采模型；分辨率相同且网格一致时直接复用一采 latent。
- `H3 自动导演｜双采样分辨率`：输出两阶段对齐后的尺寸和分辨率配置。
- `H3 自动导演｜一采/二采模型加载`、`H3 自动导演｜多模态参考模型加载`、`H3 自动导演｜动作迁移模型加载`：选择 H3 模型，可选 FL2VA/Ref2VA 混合。
- `H3 自动导演｜AV 解码`、`H3 视频迁移｜按策略解码`、`H3 TTS｜仅解码音频`、`H3 自动导演｜保存片段`：解码和保存生成结果。
- `H3 自动导演｜拼接最终视频`、`H3 TTS｜拼接最终音频`：从磁盘片段完成最终输出。
- `H3 自动导演｜音频采样切换`、`H3 自动导演｜应用音频采样配置`：提供 H3 音频采样方法、视频/音频偏移和基础 `SIGMAS`；不控制采样步数。

本版本已经移除插件自带的视频超分、补帧、独立视频加载和独立视频保存节点。旧工作流若仍包含这些节点，需要改用其他视频处理插件或删除旧节点后再运行。

## 安装与依赖

将目录放入：

```text
ComfyUI/custom_nodes/ComfyUI-H3-Auto-Director
```

重启实际运行工作流的 ComfyUI。基础功能依赖 ComfyUI H3 核心节点、H3 视频/音频 VAE 和参考素材加载节点；参考视频通常需要 VideoHelperSuite，音频加载需要当前 ComfyUI 的音频节点。

遇到节点注册、采样、张量形状或音频错误时，先将 ComfyUI 更新到最新版本并重启，再确认 H3 模型、VAE、参考素材节点和外部补丁版本一致。

## 工作流

仓库中的 `example_workflows/` 保存示例 JSON。本地工作流可以单独维护，不会在插件加载时自动覆盖。动作迁移和 TTS 工作流均为实验性工作流，不保证效果、长序列稳定性或所有版本兼容性。

基础动作迁移链路：

```text
动作迁移项目计划 -> 片段设置 -> 提示词与素材缓存 -> H3 采样/上下文 -> AV 解码 -> 保存片段 -> 拼接最终视频
```

动作迁移的参考视频、图片和音频素材都在编辑器中保存为独立记录。素材编号和素材说明是必填信息，文件路径只用于实际读取并非提示词中的编号；没有可用的多模态输入时不要臆测素材外观、服装或声音。

## 连续生成与缓存

- 片段按 1-based 序号保存到项目目录；每段会写入视频、音频和 AV latent 状态。
- 自动连续生成由项目计划和拼接控制器共同控制。关闭自动运行时保留状态，可从 `last_completed` 之后继续。
- 上下文默认读取上一段已经保存的结果，不把上一段重复写入当前片段的输出文件。自动裁剪由保存片段节点统一记录。
- 开启“所有片段完成后统一解码”时，采样阶段不做 VAE 解码，完成后由拼接节点逐段从磁盘读取 AV latent 解码并写入临时片段；该模式需要连接视频 VAE 和音频 VAE。
- 文本向量缓存可保存在硬盘。缓存记录包含提示词、素材标识、模型/编码器信息、参考图最短边模式和尺寸；缓存条件不匹配时会重新编码。
- 关闭硬盘缓存时不会写入新的文本向量文件，但已经存在的旧缓存不会自动删除。

## 音频采样

`H3 自动导演｜音频采样切换` 有两个输出：

- `采样调度信息`：连接双阶段采样的音频采样方法输入，或连接 `H3 自动导演｜应用音频采样配置`。
- `SIGMAS`：只提供基础 Sigma 端点，可连接 ComfyUI 的 `ExtendIntermediateSigmas` 或双阶段采样 Sigma 输入。步数、调度器和降噪仍由采样节点控制。

新版 ComfyUI 优先选择 `ComfyUI v0.31.0版本方法`；需要复现旧行为时选择 `ComfyUI v0.30.0版本方法`。两种模式不要同时连接，也不要再串联外部 LegacySampling 节点。旧版核心的兼容路径仍保留，但新版音视频上下文优先使用原生 AV 布局。

## 模型与 LoRA

模型加载节点只负责选择模型和可选 FL2VA/Ref2VA 混合，不负责 LoRA、注意力或显存优化。需要 LoRA、Sage Attention、Sol-Attn 或其他优化节点时，请按对应项目的接口连接；同一注意力实现通常只应启用一套。

H3 Turbo 加速 LoRA：

[aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI](https://huggingface.co/aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI)

LoRA 必须与当前 H3 模型和量化系列匹配。出现 `shape ... is invalid` 时先断开 LoRA 验证基础模型，再更换明确适配的 LoRA。

## 混合模型

启用混合时，节点以 FL2VA 为基础并覆盖 Ref2VA 的指定 AdaLN block。两份模型必须来自同一量化系列且键集合兼容；这是实验性合并，失败时应回退到纯 Ref2VA。混合加载会主动释放已加载模型，适合在工作流开始处使用。

## 姿态/深度预处理与 Union

`H3 动作迁移｜姿态/深度预处理` 可以按片段读取动作视频，选择姿态、深度或两者，并将预处理结果保存为 `control/preprocessed/segment_XXXX/` 下的视频。支持 GPU/CPU 设备、姿态类型、深度模型、分辨率匹配和可选分块处理；分辨率配置连接后优先使用第一阶段画布。

预处理视频通过 `H3_CONTROL_CONFIG` 传递路径、权重、片段和审计信息。原生 ComfyUI `MiniMaxH3Model` 不包含 Union control blocks，因此双采样节点上的元数据附加不等于 ControlNet 已介入采样；没有 VideoX-Fun 后端时，原生链路只会记录配置并继续生成。

### VideoX-Fun 专用节点（实验性）

`H3 动作迁移｜VideoX-Fun Union 专用采样` 是独立的外部管线入口，不是原生 H3 采样器的替代补丁。它要求：

1. VideoX-Fun 仓库根目录（包含 `videox_fun/` 和配置文件）；
2. 官方 Diffusers 格式的 MiniMax-H3 模型目录，包含 `transformer`、`vae`、`audio_vae`、`text_encoder`、`tokenizer`、`processor`、`scheduler` 和 `audio_scheduler`；
3. `MiniMax-H3-Fun-Controlnet-Union.safetensors` 控制权重；
4. 在 ComfyUI 使用的 Python 环境中安装 VideoX-Fun 的依赖。

该节点当前一次调用使用一种控制类型；姿态和深度需要分别运行或由外部 VideoX-Fun 工作流组合。节点的后端状态显示“已实际执行”只代表外部管线完成加载和推理。

`H3 模型格式检查` 只读取 safetensors header，用于诊断模型格式：

- Diffusers 目录会列出缺少的组件；
- ComfyUI H3 safetensors 会标记普通、INT8/ConvRot、AdaLN 曲线和控制分支信息；
- CUI 单文件不会被 VideoX-Fun 节点直接转换。尤其是 `*_pruned_int8_convrot.safetensors`、W4A4/W4A8 和包含 `adaln_t_table` 的文件只能继续用于 ComfyUI 原生 H3，或改用官方 Diffusers 基础模型。

## 解码已保存 latent

`H3 解码｜目录批量解码并保存视频` 可识别项目目录、`cache/`、`cache_stage1/` 或直接包含 AV latent 的目录。节点按片段编号排序，每次只解码一个片段并写入磁盘，可选择中间片段、文件名、最终文件名和从第二段开始的上下文裁剪帧数。输入必须是插件保存的同时包含 `video` 与 `audio` 的 H3 AV latent。

## 故障排查

- `Output file does not contain any stream`：参考视频没有音轨或音轨不可读。插件会保留画面并跳过该音轨；需要音频时请换用包含可解码音轨的视频并开启“传递视频音频”。
- `tuple index out of range` 或 conditioning pair 错误：通常是旧缓存或外部节点返回了非标准 `CONDITIONING` 结构。插件会跳过无效条目并保留原条件；仍失败时清理当前项目缓存后重跑。
- 音频条件行数不一致：确认所有片段使用同一 ComfyUI 核心版本，重新执行当前片段的参考缓存，并优先使用新版 AV 音频采样方法。
- 控制预处理审计失败：检查控制视频帧数是否为 `17*n+5`、宽高是否为 32 的倍数、像素范围是否为 0..1。
- `video is not a valid path`：在项目编辑器中重新上传素材并保存，确保计划中的路径指向 `input` 目录内的实际文件。

## 来源与鸣谢

- [ComfyUI-MiniMax-H3-Hybrid](https://github.com/ANe5s/ComfyUI-MiniMax-H3-Hybrid)：H3 混合模型加载思路。
- [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)：连续视频上下文设计参考。
- [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)：H3 latent 放大参考。
- [MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) 与 [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)：Union 控制分支和外部管线。
- [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)：DWPose、OpenPose 和深度预处理适配参考。
- [ComfyUI-MiniMax-H3-LegacySampling](https://github.com/starsFriday/ComfyUI-MiniMax-H3-LegacySampling)：旧版音频采样兼容思路。

第三方代码、模型和依赖各自遵循上游许可证。本项目代码以 GPL-3.0-or-later 发布，详见 `LICENSE`。
