# Third-Party Notices

## ComfyUI-H3-Motion-Context

The sequential video-context design and compatibility behavior in this project
were informed by [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
by NikoDemon80. The current Auto Director repository uses ComfyUI's native H3
guide path when available and includes an independent compatibility layer, so
the external node is optional for the bundled workflows. This repository does
not vendor that project's source code. Follow its GNU General Public License
v3.0 when using or redistributing the upstream project.

## MiniMax H3 and ComfyUI

MiniMax H3 model files, ComfyUI core nodes, VideoHelperSuite, ffmpeg, and other
dependencies are separately distributed projects. They are not included here;
their respective licenses and terms apply.

## MiniMax H3 Legacy Sampling implementation

The optional `ComfyUI v0.30.0版本方法` branch in
`H3 自动导演｜音频采样切换` is derived from the compatibility technique published
in [ComfyUI-MiniMax-H3-LegacySampling](https://github.com/starsFriday/ComfyUI-MiniMax-H3-LegacySampling)
by starsFriday. The implementation is included directly in this repository so
that the external node is not required. Please keep the upstream attribution
and consult that project's license and terms when redistributing derivative
work.

## ComfyUI-MiniMax-H3-Hybrid

The optional H3 hybrid-model loading path is informed by the model-combination
approach documented in [ComfyUI-MiniMax-H3-Hybrid](https://github.com/ANe5s/ComfyUI-MiniMax-H3-Hybrid)
by ANe5s. We thank ANe5s for the upstream investigation. This repository uses
an independent implementation and does not vendor that project's source code;
retain the upstream attribution and follow its license and terms when using or
redistributing related work.

## ComfyUI-CustomNodeKit

Advanced color-drift correction ideas were evaluated against
[ComfyUI-CustomNodeKit](https://github.com/user2318/ComfyUI-CustomNodeKit). The
H3 implementation is an independent, conservative adaptation with scene-cut
protection; the original project and its license remain acknowledged here.

## Comfyui_Minimax_h3_latent_Upscaler

The optional H3 latent upscaling path is informed by
[Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
by LBH-123-AI. This repository contains an independent integration and does
not redistribute the upstream latent upscaling model. Refer to the upstream
README for model download links, file names, installation instructions, and
license terms.
