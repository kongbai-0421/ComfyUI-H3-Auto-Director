---
name: h3-auto-director-json
description: Generate strict JSON segment plans for the MiniMax H3 Auto Director ComfyUI plugin. Use when a user asks for H3 multi-segment prompts, durations, continuation settings, audio restart points, or multimodal reference assignments. Default to no references and output only a valid JSON array.
---

# H3 Auto Director JSON

Generate the `segments_json` value consumed by `H3AutoDirectorPlan`. Follow the
MiniMax H3 official full-reference prompt format, but return only the JSON
array. Do not add Markdown fences, explanations, headings, or comments outside
the JSON.

## Output Contract

Return a non-empty JSON array. Every array item must contain exactly these
plugin-facing fields unless the user explicitly requests another supported
field:

```json
{
  "prompt": "subject_definitions:\n...\n\nsummary:\n...\n\nretention_analysis:\n...\n\ndetailed_description:\n...\n\noverall_soundscape:\n...\n\nnon_diegetic_music:\n...",
  "duration": 4,
  "audio_restart": false,
  "continue_video": false,
  "references": []
}
```

Use these defaults:

- `duration`: `5` unless the user gives a duration. Keep every value between `4` and `15` seconds.
- `audio_restart`: `false`; set `true` only at a user-requested or clearly justified audio reset point.
- `continue_video`: `false` for the first segment; `true` for later segments unless the user requests independent scenes or disables continuation.
- `references`: use `[]` unless the user supplies a local file path for the asset. Never invent filenames, paths, URLs, reference images, videos, or audio.
- Image references have no independent duration. Never emit `duration` or `image_duration` inside an image reference; the segment-level `duration` controls the complete generated clip. Video and audio metadata may retain `duration` only for display or source-media bookkeeping.

The plugin's reference arrays and ComfyUI sockets are zero-based, but MiniMax H3
prompt labels are one-based. Only when the user supplies a local file path,
preserve that exact path and assign prompt labels by type: `<Picture 1>`,
`<Picture 2>`, `<Video 1>`, `<Audio 1>`. Image and video indices are
independent. A video reference's embedded soundtrack is detected by the plugin
automatically and is enabled by default. Set `video_audio_enabled` to `false`
on that video reference only when the user asks to pass video frames without its
soundtrack; do not invent a `has_audio` field because detection is automatic.
Audio labels follow H3 presentation order: enabled video soundtracks first, then
standalone audio references. Disabling a video soundtrack removes its `<Audio N>`
label and shifts later standalone audio labels down. If the user provides no local file path, every
segment's `references` must remain `[]`. Do not create a reference entry from
a visual description, an attachment name, “reference” wording, an uploaded
preview, a URL, or an inferred asset mapping.

## Prompt Requirements

Write every `prompt` in English except dialogue, lyrics, and visible scene text,
which must stay in their original language. Use the six sections in this exact
order:

1. `subject_definitions`
2. `summary`
3. `retention_analysis`
4. `detailed_description`
5. `overall_soundscape`
6. `non_diegetic_music`

Use the official Ref2VA relationship markers in `retention_analysis`:
`fully_preserved`, `partially_preserved`, `attribute_transfer`, or
`weak_reference`. Use `fully_copy`, `partially_copy`, `reference`, or
`weak_reference` for audio entries.

If the user provides no local file paths, define the target subjects directly without
inventing `<Picture>`, `<Video>`, or `<Audio>` labels. If references exist,
define each label before using it, keep its meaning stable in all six sections,
and state precisely what is preserved versus transferred.

## Segment Detail Rule

Do not reduce a segment to a short action label such as “the character runs” or
“the character walks”, unless the user explicitly requests that minimal
description. Each ordinary segment must describe a complete mini-shot:

- composition, framing, subject position, and environment;
- the subject's appearance and continuity from the previous segment;
- an ordered action progression with at least two or three state changes;
- body direction, weight shift, hand/leg motion, and interaction with props when relevant;
- camera type, movement direction, amplitude, speed, and the intended ending pose;
- physical sound events that correspond to the visible motion;
- lighting, atmosphere, and important background continuity.

For a 4-6 second segment, describe an opening state, a developing action, a
peak or transition, and a final state. Use natural timing phrases such as
`during the first second`, `then`, `midway through the shot`, and `end on`.
Do not invent extra plot events merely to make a prompt longer. Preserve the
user's requested action, but expand its visual and temporal execution.

Every segment must be self-contained enough to run by itself while explicitly
continuing the previous segment when `continue_video` is true. The first
sentence of a continued segment should identify the previous ending state and
the next motion. Do not make all segments identical: vary the shot, movement
phase, camera behavior, and sound according to the user's story or supplied
timeline.

## Continuity and Audio

Keep identity, costume, props, lighting direction, environment, and camera
logic consistent across segments unless the user requests a change. Use
`audio_restart: true` at a deliberate musical, scene, or sound-design boundary;
otherwise describe continuous ambience and music across the cut.

When the user supplies no local audio file path, generate `overall_soundscape` and
`non_diegetic_music` descriptions but keep `references` empty. Do not add an
audio reference just because the prompt mentions music or sound effects.

## Validation Before Returning

Before emitting the result:

1. Confirm the result parses as one JSON array.
2. Confirm every segment has a non-empty `prompt`, valid duration, boolean audio and continuation flags, and a list-valued `references` field.
3. Confirm `references` is `[]` for every segment unless the user supplied local file paths. Confirm every non-empty reference has an exact user-provided local file path and a valid type (`image`, `video`, or `audio`). For video references, `video_audio_enabled` is optional and must be boolean when present; omit it for the default enabled behavior.
4. Confirm no segment exceeds 12 total references, 9 images, 3 videos, or 3 independent audios.
5. Confirm every prompt has all six sections in the required order.
6. Confirm no prose, Markdown fence, or trailing comma appears outside the JSON.

If the user supplies only a story idea and no segment count, infer a sensible
number of 4-6 second segments from the requested duration, but do not add
references. If the user explicitly asks for a deliberately simple action, obey
that request while still identifying the shot, subject, camera, timing, and
ending state.
