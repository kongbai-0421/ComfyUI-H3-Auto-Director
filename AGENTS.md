# H3 Auto Director Project Context

- This is the MiniMax H3 ComfyUI automation plugin. Its upstream repository is https://github.com/kongbai-0421/ComfyUI-H3-Auto-Director.
- The reusable JSON prompt skill is installed at `%USERPROFILE%\.codex\skills\h3-auto-director-json` and is also bundled under `skills/h3-auto-director-json`.
- In the one-shot prompt embedding cache, encode all segment reference assets with the VAE first, then encode every segment's text continuously, and unload the CLIP text encoder only after the full batch completes.
- Keep ordinary per-segment encoding behavior unchanged unless the task explicitly changes it.
- H3 prompt JSON must leave `references` empty unless the user supplies an exact local asset file path. Never invent paths, URLs, filenames, or labels.
- H3 prompt reference labels are 1-based (`<Picture 1>`, `<Video 1>`, `<Audio 1>`); only internal ComfyUI socket names remain 0-based (`ref_image_0`, etc.). Audio labels follow H3 presentation order.
- Do not add an `image_duration` segment field: H3 reference images condition the entire clip and do not accept a per-image duration.
- Context color correction belongs in the save-segment path: support a fixed segment-1 anchor or the immediately previous segment, apply one frame transform before writing the context video, and never rewrite the paired AV latent or audio stream.
- Python node changes require restarting the ComfyUI instance that actually loads this custom node.
- Reference uploads support a persisted default-path switch and Python/browser picker mode. Python mode opens the native Windows dialog on the ComfyUI machine and imports into `input/h3_refs/*`; keep the browser picker as a fallback.
- Do not modify example or user workflows unless the user explicitly requests workflow changes.
