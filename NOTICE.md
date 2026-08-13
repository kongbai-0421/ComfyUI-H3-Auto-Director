# Third-Party Notices

## ComfyUI-H3-Motion-Context

The sequential video-context capability in this project integrates with and
depends on [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
by NikoDemon80. It provides the H3 motion-context node used to carry prior
video frames and AV latent state into the following segment.

This repository does not vendor that project's source code. Install it
separately and comply with its GNU General Public License v3.0.

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
